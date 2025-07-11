import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch
from datasets import DatasetDict
from transformers import AutoTokenizer

import common.utils.logging as logging_utils
from utils.data_utils import HamiltonianCycleDatasetLoader
from utils.data_utils import get_preference_dataset_loader
from utils.strings import DEFAULT_TRAIN_SPLIT_NAME, DEFAULT_TEST_SPLIT_NAME


def __filter_by_length(dataset, tokenizer, max_prompt_length: int = -1, max_response_length: int = -1):
    def filter_func(example):
        prompt_not_too_long = True
        if max_prompt_length > 0:
            prompt_length = len(tokenizer.encode(example["prompt"]))
            prompt_not_too_long = prompt_length <= max_prompt_length

        responses_not_too_long = True
        if max_response_length > 0:
            chosen_length = len(tokenizer.encode(example["chosen"]))
            rejected_length = len(tokenizer.encode(example["rejected"]))
            responses_not_too_long = chosen_length <= max_response_length and rejected_length <= max_response_length

        return prompt_not_too_long and responses_not_too_long

    return dataset.filter(filter_func)


def __log_dataset_properties(train_dataset, test_dataset, tokenizer=None):
    logging_utils.info("=" * 110)
    logging_utils.info("Dataset Characteristics")
    logging_utils.info("-" * 110)
    for split_name, dataset in zip(["train", "test"], [train_dataset, test_dataset]):
        logging_utils.info(f"Split: {split_name}")
        logging_utils.info(f"Number of samples: {dataset.num_rows}")
        if tokenizer:
            logging_utils.info(f"Mean prompt length: {np.mean([len(tokenizer.encode(x['prompt'])) for x in dataset])}")
            logging_utils.info(f"Mean chosen response length: {np.mean([len(tokenizer.encode(x['chosen'])) for x in dataset])}")
            logging_utils.info(f"Mean rejected response length: {np.mean([len(tokenizer.encode(x['rejected'])) for x in dataset])}")
        logging_utils.info("-" * 110)

    logging_utils.info("=" * 110)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of the dataset. Needs to correspond to dataset supported in preference_dataset_loaders.py")
    parser.add_argument("--cache_dir", type=str, help="Hugging Face cache dir.")
    parser.add_argument("--random_seed", type=int, default=982, help="Random seed for data preparation.")
    parser.add_argument("--output_dir", type=str, default="data_files", help="Path to a directory for saving the dataset in.")

    parser.add_argument("--num_train_samples", type=int, default=2000, help="Number of training samples to use.")
    parser.add_argument("--num_test_samples", type=int, default=200, help="Number of test samples to use.")
    parser.add_argument("--tokenizer_for_length_filtering", type=str, default="meta-llama/Llama-3.2-1B",
                        help="Tokenizer for length filtering. If None given, will not filter out samples.")
    parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="Samples with prompts longer than this will be filtered out.")
    parser.add_argument("--max_response_length", type=int, default=512,
                        help="Samples in which one of the responses is longer than this will be filtered out.")
    parser.add_argument("--filter_out_translate_prompts", action="store_true",
                        help="If True, will filter out samples with prompts that contain 'translate' or 'translation' (case insensitive) in them.")

    parser.add_argument("--ham_cycle_num_vertices", type=int, default=10,
                        help="Number of vertices in graphs to generate for the "
                             "Hamiltonian cycle task. Only relevant if dataset is 'hamiltonian_cycle'.")
    parser.add_argument("--ham_cycle_edge_p", type=float, default=0.05,
                        help="Probability of adding an edge between two vertices in the graph for the Hamiltonian cycle task. "
                             "Only relevant if dataset is 'hamiltonian_cycle'.")

    args = parser.parse_args()
    logging_utils.init_console_logging()
    str_timestamp = datetime.now(timezone.utc).strftime("%Y_%m_%d-%H_%M_%S")
    filter_out_translate_str = "_no_translate_prompts" if args.filter_out_translate_prompts else ""
    dataset_str = args.dataset if args.dataset != "hamiltonian_cycle" else f"{args.dataset}_v{args.ham_cycle_num_vertices}_p{args.ham_cycle_edge_p}"
    path_for_saving = (f"{args.output_dir}/{dataset_str}_seed_{args.random_seed}_ntrain_{args.num_train_samples}"
                       f"_ntest_{args.num_test_samples}{filter_out_translate_str}_{str_timestamp}")
    logging_utils.init_file_logging(log_file_name_prefix=f"pref_data", output_dir=path_for_saving)

    os.makedirs(path_for_saving, exist_ok=True)
    with open(os.path.join(path_for_saving, "run_config.json"), "w") as file:
        json.dump(args.__dict__, file, indent=2)

    rnd_gen = torch.Generator().manual_seed(args.random_seed) if args.random_seed > 0 else None
    kwargs = {} if args.dataset != "hamiltonian_cycle" else {"num_vertices": args.ham_cycle_num_vertices, "edge_p": args.ham_cycle_edge_p}
    train_dataset, test_dataset = get_preference_dataset_loader(args.dataset).load_and_preprocess_preference_dataset(cache_dir=args.cache_dir,
                                                                                                                     rnd_gen=rnd_gen,
                                                                                                                     **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_for_length_filtering,
                                              use_fast=True,
                                              trust_remote_code=True,
                                              cache_dir=args.cache_dir) if args.tokenizer_for_length_filtering else None
    if args.dataset == "hamiltonian_cycle":
        tokenizer.add_tokens([HamiltonianCycleDatasetLoader.SEP_TOKEN, HamiltonianCycleDatasetLoader.EDGE_SEP_TOKEN])

    logging_utils.info("Dataset characteristics before length filtering")
    __log_dataset_properties(train_dataset, test_dataset, tokenizer=tokenizer)

    # Filter out samples with too long prompts or too long responses
    if args.tokenizer_for_length_filtering:
        train_dataset = __filter_by_length(train_dataset, tokenizer, args.max_prompt_length, args.max_response_length)
        test_dataset = __filter_by_length(test_dataset, tokenizer, args.max_prompt_length, args.max_response_length)
        logging_utils.info("Dataset characteristics after length filtering")
        __log_dataset_properties(train_dataset, test_dataset, tokenizer=tokenizer)

    # Filter out samples with prompts that contain "translate" or "translation" in them
    if args.filter_out_translate_prompts:
        train_dataset = train_dataset.filter(lambda example: "translate" not in example["prompt"].lower()
                                                             and "translation" not in example["prompt"].lower())
        test_dataset = test_dataset.filter(lambda example: "translate" not in example["prompt"].lower()
                                                           and "translation" not in example["prompt"].lower())
        logging_utils.info("Dataset characteristics after filtering out prompts that contain 'translate' or 'translation' in them")
        __log_dataset_properties(train_dataset, test_dataset, tokenizer=tokenizer)

    num_train_samples = min(len(train_dataset), args.num_train_samples) if args.num_train_samples > 0 else len(train_dataset)
    train_perm = torch.randperm(len(train_dataset), generator=rnd_gen)
    train_dataset = train_dataset.select(train_perm[:num_train_samples])

    num_test_samples = min(len(test_dataset), args.num_test_samples) if args.num_test_samples > 0 else len(test_dataset)
    test_perm = torch.randperm(len(test_dataset), generator=rnd_gen)
    test_dataset = test_dataset.select(test_perm[:num_test_samples])

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
        logging_utils.exception("Exception while running preference data creation script.")
        raise
    finally:
        end_time = datetime.now(timezone.utc)
        logging_utils.info(f"Finished preference dataset creation script. Time took: {end_time - start_time}")
