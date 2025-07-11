import argparse
import os
from datetime import datetime, timezone

import jsonlines
import torch
from datasets import load_from_disk
from openai import OpenAI

import common.utils.logging as logging_utils


def __create_batch_openai_rewrite_call(output_dir: str, train_dataset, test_dataset, prompt_format: str, openai_model: str, max_tokens: int = 512):
    client = OpenAI()
    all_examples = list(train_dataset) + list(test_dataset)

    request_jsons = []
    for i, example in enumerate(all_examples):
        for j, field in enumerate(["chosen", "rejected"]):
            request_order = i * 2 + j
            request_jsons.append({
                "custom_id": f"{request_order}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": openai_model,
                    "messages": [{"role": "user", "content": prompt_format.format(response=example[field])}],
                    "max_tokens": max_tokens
                }
            })

    requests_jsonl_path = os.path.join(output_dir, "requests.jsonl")
    with jsonlines.open(requests_jsonl_path, mode='w') as writer:
        for request in request_jsons:
            writer.write(request)

    logging_utils.info(f"Saved requests jsonl at: {requests_jsonl_path}")

    # Upload batch input jsonl
    batch_input_file = client.files.create(file=open(requests_jsonl_path, "rb"), purpose="batch")
    batch_input_file_id = batch_input_file.id
    logging_utils.info(f"Requests jsonl uploaded to OpenAI and was given the ID: {batch_input_file_id}")

    # Create the batch job
    batch_obj = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "Batch job for rewriting responses"}
    )

    return batch_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the preference dataset on disk. The dataset should contain train and test splits with examples having fields "
                             "'prompt', 'chosen', 'rejected'.")
    parser.add_argument("--prompt_format", type=str, required=True,
                        help="Format of the prompt. Should include a placeholder for the response to write of the form '{response}'.")
    parser.add_argument("--rewrite_type", type=str, default="", help="Optional name for the rewrite type done according to the prompt.")
    parser.add_argument("--random_seed", type=int, default=517, help="Random seed for data selection.")
    parser.add_argument("--output_dir", type=str, default="outputs/rewrite_calls", help="Path to a directory for saving logs.")
    parser.add_argument("--num_train_samples", type=int, default=200, help="Number of training samples to rewrite responses for.")
    parser.add_argument("--num_test_samples", type=int, default=200, help="Number of test samples to rewrite responses for.")
    parser.add_argument("--max_response_length", type=int, default=512, help="Max tokens for the response.")
    parser.add_argument("--openai_model", type=str, default="gpt-4.1-2025-04-14", help="Name of the OpenAI model to use.")

    args = parser.parse_args()
    logging_utils.init_console_logging()
    str_timestamp = datetime.now(timezone.utc).strftime("%Y_%m_%d-%H_%M_%S")
    dir_for_saving = (f"{args.output_dir}/{args.rewrite_type}_{args.openai_model}_seed_{args.random_seed}_ntrain_{args.num_train_samples}"
                      f"_ntest_{args.num_test_samples}_{str_timestamp}")
    logging_utils.init_file_logging(log_file_name_prefix="log", output_dir=dir_for_saving)

    preference_dataset = load_from_disk(args.dataset_path)
    train_dataset, test_dataset = preference_dataset["train"], preference_dataset["test"]
    rnd_gen = torch.Generator().manual_seed(args.random_seed) if args.random_seed > 0 else None

    num_train_samples = min(len(train_dataset), args.num_train_samples) if args.num_train_samples > 0 else len(train_dataset)
    train_perm = torch.randperm(len(train_dataset), generator=rnd_gen)
    train_sample_indices = torch.sort(train_perm[:num_train_samples]).values
    train_dataset = train_dataset.select(train_sample_indices)

    num_test_samples = min(len(test_dataset), args.num_test_samples) if args.num_test_samples > 0 else len(test_dataset)
    test_perm = torch.randperm(len(test_dataset), generator=rnd_gen)
    test_sample_indices = torch.sort(test_perm[:num_test_samples]).values
    test_dataset = test_dataset.select(test_sample_indices)

    batch_obj = __create_batch_openai_rewrite_call(dir_for_saving, train_dataset, test_dataset,
                                                   prompt_format=args.prompt_format,
                                                   openai_model=args.openai_model,
                                                   max_tokens=args.max_response_length)

    rewrite_call_details = args.__dict__
    rewrite_call_details.update({
        "train_sample_indices": train_sample_indices,
        "test_sample_indices": test_sample_indices,
        "openai_model": args.openai_model,
        "prompt_format": args.prompt_format,
        "batch": batch_obj
    })

    os.makedirs(dir_for_saving, exist_ok=True)
    torch.save(rewrite_call_details, os.path.join(dir_for_saving, "rewrite_call_details.pt"))
    logging_utils.info(f"Rewrite call details saved at: {os.path.join(dir_for_saving, 'rewrite_call_details.pt')}\n"
                       f"Batch ID: {batch_obj.id}\n"
                       f"Full details: {rewrite_call_details}")


if __name__ == "__main__":
    start_time = datetime.now(timezone.utc)
    try:
        main()
    except Exception:
        logging_utils.exception("Exception while creating rewrite response call script.")
        raise
    finally:
        end_time = datetime.now(timezone.utc)
        logging_utils.info(f"Finished response rewriting call script. Time took: {end_time - start_time}")
