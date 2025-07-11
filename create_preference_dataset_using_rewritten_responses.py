import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch
from datasets import load_from_disk, DatasetDict, Dataset
from openai import OpenAI
from transformers import AutoTokenizer

import common.utils.logging as logging_utils
from utils.strings import DEFAULT_TRAIN_SPLIT_NAME, DEFAULT_TEST_SPLIT_NAME


def __log_dataset_properties(train_dataset, test_dataset, tokenizer=None):
    logging_utils.info("=" * 110)
    logging_utils.info("Dataset Characteristics")
    logging_utils.info("-" * 110)
    for split_name, dataset in zip(["train", "test"], [train_dataset, test_dataset]):
        logging_utils.info(f"Split: {split_name}")
        logging_utils.info(f"Number of samples: {dataset.num_rows}")
        logging_utils.info(f"Mean chosen response length: {np.mean([len(tokenizer.encode(x['chosen'])) for x in dataset])}")
        logging_utils.info(f"Mean rejected response length: {np.mean([len(tokenizer.encode(x['rejected'])) for x in dataset])}")
        logging_utils.info("-" * 110)

    logging_utils.info("=" * 110)


def __get_responses_and_model_name_from_openai_batch_request(batch_id: str):
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    if batch.error_file_id:
        error_file = client.files.content(batch.error_file_id)
        logging_utils.error(f"Errors occurred in the batch API requests.\n"
                            f"Details: {error_file.text.strip()}")
        raise RuntimeError("Errors occurred in the batch API requests.")

    batch_output = client.files.content(batch.output_file_id)
    api_responses = [json.loads(line) for line in batch_output.text.strip().split('\n')]
    int_id_to_response_body = {int(api_response["custom_id"]): api_response["response"]["body"] for api_response in api_responses}

    sorted_response_bodies = [int_id_to_response_body[i] for i in range(len(int_id_to_response_body))]
    model_name = sorted_response_bodies[0]["model"]

    train_responses_list = [response_body["choices"][0]["message"]["content"]
                            for response_body in sorted_response_bodies[:len(sorted_response_bodies) // 2]]
    train_responses = []
    for i in range(0, len(train_responses_list), 2):
        train_responses.append({"chosen": train_responses_list[i], "rejected": train_responses_list[i + 1]})

    test_responses_list = [response_body["choices"][0]["message"]["content"]
                           for response_body in sorted_response_bodies[len(sorted_response_bodies) // 2:]]
    test_responses = []
    for i in range(0, len(test_responses_list), 2):
        test_responses.append({"chosen": test_responses_list[i], "rejected": test_responses_list[i + 1]})

    return train_responses, test_responses, model_name


def __create_dataset_with_rewritten_responses(dataset, new_responses_dict, orig_sample_indices):
    new_examples = []
    for i, example in enumerate(dataset):
        new_examples.append({
            "orig_index": orig_sample_indices[i].item(),
            "prompt": example["prompt"],
            "chosen": new_responses_dict[i]["chosen"],
            "rejected": new_responses_dict[i]["rejected"],
            "orig_chosen": example["chosen"],
            "orig_rejected": example["rejected"]
        })

    return Dataset.from_list(new_examples)


def __log_token_overlap_stats(dataset, split_name: str, tokenizer):
    logging_utils.info("=" * 110)
    logging_utils.info(f"Token overlap statistics for {split_name} split")
    logging_utils.info("-" * 110)

    overlaps_of_orig_responses_with_themselves = []
    overlaps_of_new_responses_with_prev = []
    for example in dataset:
        orig_chosen_tokens = torch.tensor(tokenizer.encode(example["orig_chosen"]))
        orig_rejected_tokens = torch.tensor(tokenizer.encode(example["orig_rejected"]))
        new_chosen_tokens = torch.tensor(tokenizer.encode(example["chosen"]))
        new_rejected_tokens = torch.tensor(tokenizer.encode(example["rejected"]))

        orig_chosen_with_self_overlap = (orig_chosen_tokens.unsqueeze(dim=1) == orig_chosen_tokens).sum() / (orig_chosen_tokens.numel() ** 2)
        orig_rejected_with_self_overlap = (orig_rejected_tokens.unsqueeze(dim=1) == orig_rejected_tokens).sum() / (orig_rejected_tokens.numel() ** 2)
        overlaps_of_orig_responses_with_themselves.extend([orig_chosen_with_self_overlap.item(), orig_rejected_with_self_overlap.item()])

        orig_chosen_with_new_overlap = ((orig_chosen_tokens.unsqueeze(dim=1) == new_chosen_tokens).sum() /
                                        (orig_chosen_tokens.numel() * new_chosen_tokens.numel()))
        orig_rejected_with_new_overlap = ((orig_rejected_tokens.unsqueeze(dim=1) == new_rejected_tokens).sum() /
                                          (orig_rejected_tokens.numel() * new_rejected_tokens.numel()))
        overlaps_of_new_responses_with_prev.extend([orig_chosen_with_new_overlap.item(), orig_rejected_with_new_overlap.item()])

    overlaps_of_prev_responses_with_themselves_mean = np.mean(overlaps_of_orig_responses_with_themselves)
    overlaps_of_new_responses_with_prev_mean = np.mean(overlaps_of_new_responses_with_prev)
    logging_utils.info(f"Mean token overlap of original responses with themselves: {overlaps_of_prev_responses_with_themselves_mean}")
    logging_utils.info(f"Mean token overlap of new responses with original responses: {overlaps_of_new_responses_with_prev_mean}")
    logging_utils.info("=" * 110)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite_details_path", type=str, required=True,
                        help="Path to the rewrite details file created when calling the OpenAI API.")
    parser.add_argument("--output_dir", type=str, default="data_files/rewritten", help="Path to a directory for saving the datasets.")
    parser.add_argument("--cache_dir", type=str, help="Hugging Face cache dir.")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-1B",
                        help="Optional tokenizer name to use for printing statistics on response lengths.")
    args = parser.parse_args()

    rewrite_details = torch.load(args.rewrite_details_path)
    orig_dataset_path = rewrite_details['dataset_path']
    orig_dataset_dir = os.path.basename(orig_dataset_path)

    logging_utils.init_console_logging()
    str_timestamp = datetime.now(timezone.utc).strftime("%Y_%m_%d-%H_%M_%S")
    path_for_saving = f"{args.output_dir}/{rewrite_details['rewrite_type']}_{rewrite_details['openai_model']}_{orig_dataset_dir}_{str_timestamp}"
    logging_utils.init_file_logging(log_file_name_prefix="log", output_dir=path_for_saving)
    logging_utils.info("Starting to create preference dataset with rewritten responses based on arguments:\n"
                       f"{args.__dict__}")

    train_responses, test_responses, model_name = __get_responses_and_model_name_from_openai_batch_request(rewrite_details["batch"].id)
    logging_utils.info(f"Fetched responses from OpenAI API, full name of the model used: {model_name}")

    # Load original dataset
    preference_dataset = load_from_disk(orig_dataset_path)
    train_dataset, test_dataset = preference_dataset["train"], preference_dataset["test"]
    train_sample_indices = rewrite_details["train_sample_indices"]
    test_sample_indices = rewrite_details["test_sample_indices"]
    train_dataset = train_dataset.select(train_sample_indices)
    test_dataset = test_dataset.select(test_sample_indices)

    # Create new dataset with rewritten responses, keeping originals as well
    train_dataset = __create_dataset_with_rewritten_responses(train_dataset, train_responses, train_sample_indices)
    test_dataset = __create_dataset_with_rewritten_responses(test_dataset, test_responses, test_sample_indices)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                              use_fast=True,
                                              trust_remote_code=True,
                                              cache_dir=args.cache_dir) if args.tokenizer else None
    if tokenizer:
        __log_token_overlap_stats(train_dataset, split_name=DEFAULT_TRAIN_SPLIT_NAME, tokenizer=tokenizer)
        __log_token_overlap_stats(test_dataset, split_name=DEFAULT_TEST_SPLIT_NAME, tokenizer=tokenizer)

    logging_utils.info("Final dataset characteristics")
    __log_dataset_properties(train_dataset, test_dataset, tokenizer=tokenizer)
    dataset = DatasetDict({
        DEFAULT_TRAIN_SPLIT_NAME: train_dataset,
        DEFAULT_TEST_SPLIT_NAME: test_dataset
    })
    dataset.save_to_disk(path_for_saving)
    logging_utils.info(f"Dataset saved locally at: {path_for_saving}")


if __name__ == "__main__":
    start_time = datetime.now(timezone.utc)
    try:
        main()
    except Exception:
        logging_utils.exception("Exception while creating datasets with rewritten responses.")
        raise
    finally:
        end_time = datetime.now(timezone.utc)
        logging_utils.info(f"Finished dataset with rewritten responses creation script. Time took: {end_time - start_time}")
