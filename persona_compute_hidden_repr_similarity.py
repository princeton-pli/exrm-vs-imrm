import argparse
import itertools
import os
from datetime import datetime
from typing import List

import numpy as np
import torch
from cvxopt import matrix, solvers
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import common.utils.logging as logging_utils
from rm_exps.data.persona_controlled_datamodule import PersonaControlledDataModule
from utils.rm_utils import (
    compute_hidden_representations_with_explicit_rm,
    convert_preference_batch_to_chat_format,
    tokenize_and_prepare_concatenated_preference_batch,
)
from utils.sharedmisc import (
    update_tokenizer,
    update_model_num_embeddings_and_special_tokens,
)


def __load_tokenizer_and_model(logger, model_name, cache_dir: str = None, device=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name,
        num_labels=1,
        device_map=device,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir
    )
    tokenizer, num_added_tokens = update_tokenizer(tokenizer, logger=logger)

    if num_added_tokens > 0:
        logger.warning(
            "Updating model embeddings since tokens were added to the tokenizer."
        )
        model = update_model_num_embeddings_and_special_tokens(model, tokenizer)

    if device is not None:
        model.to(device)

    return tokenizer, model


def __compute_hidden_representations_for_batch(batch, model, tokenizer, device):
    batch = convert_preference_batch_to_chat_format(tokenizer, batch)
    prompts = batch["prompt"]
    chosen_responses = batch["chosen"]
    rejected_responses = batch["rejected"]

    input_ids, attention_mask, loss_mask = tokenize_and_prepare_concatenated_preference_batch(tokenizer, prompts, chosen_responses,
                                                                                              rejected_responses, device)

    all_hidden_representations = compute_hidden_representations_with_explicit_rm(model, input_ids, attention_mask)
    return all_hidden_representations


def __compute_yes_minus_no_hidden_representations_for_seen_responses(dataloader, model, tokenizer, device):
    yes_minus_no_seen_hidden_representations = []
    for batch in dataloader:
        all_hidden_representations = __compute_hidden_representations_for_batch(batch, model, tokenizer, device)
        chosen_hidden_repr = all_hidden_representations[: all_hidden_representations.shape[0] // 2].detach()
        rejected_hidden_repr = all_hidden_representations[all_hidden_representations.shape[0] // 2:].detach()
        yes_minus_no_seen_hidden_representations.append(chosen_hidden_repr - rejected_hidden_repr)

    yes_minus_no_seen_hidden_representations = torch.cat(yes_minus_no_seen_hidden_representations)
    return yes_minus_no_seen_hidden_representations


def __modify_dataset_for_token(dataset, prefix: str, token: str, is_yes_token: bool):
    example_list = []

    for i, example in enumerate(dataset):
        new_example = example.copy()
        response_name = "chosen" if is_yes_token else "rejected"
        new_example[response_name] = prefix + token
        example_list.append(new_example)

    return Dataset.from_list(example_list)


def __compute_hidden_representations_for_token(model, tokenizer, prefix: str, token: str, dataset, is_yes_token: bool, batch_size: int, device=None):
    dataset = __modify_dataset_for_token(dataset, prefix, token, is_yes_token=is_yes_token)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    hidden_representations_list = []
    for batch in dataloader:
        hidden_representations = __compute_hidden_representations_for_batch(batch, model, tokenizer, device)
        hidden_representations = hidden_representations[: hidden_representations.shape[0] // 2].detach() \
            if is_yes_token else hidden_representations[hidden_representations.shape[0] // 2:].detach()
        hidden_representations_list.append(hidden_representations)

    return torch.cat(hidden_representations_list)


def __compute_yes_minus_no_hidden_representations_for_unseen_responses(
        dataset,
        model,
        tokenizer,
        eval_output_tokens_matching_yes: List[str],
        eval_output_tokens_matching_no: List[str],
        eval_prefix_matching_yes: str,
        eval_prefix_matching_no: str,
        batch_size: int,
        device
):
    per_yes_response_hidden_repr = {}
    for token in eval_output_tokens_matching_yes:
        yes_response_hidden_repr = __compute_hidden_representations_for_token(
            model,
            tokenizer,
            eval_prefix_matching_yes,
            token,
            dataset,
            is_yes_token=True,
            batch_size=batch_size,
            device=device
        )
        per_yes_response_hidden_repr[token] = yes_response_hidden_repr

    per_no_response_hidden_repr = {}
    for token in eval_output_tokens_matching_no:
        no_response_hidden_repr = __compute_hidden_representations_for_token(
            model,
            tokenizer,
            eval_prefix_matching_no,
            token,
            dataset,
            is_yes_token=False,
            batch_size=batch_size,
            device=device
        )
        per_no_response_hidden_repr[token] = no_response_hidden_repr

    yes_minus_no_hidden_representations_list = []
    for unseen_yes_response_hidden_repr, unseen_no_response_hidden_repr in itertools.product(per_yes_response_hidden_repr.values(),
                                                                                             per_no_response_hidden_repr.values()):
        yes_minus_no_hidden_repr = unseen_yes_response_hidden_repr - unseen_no_response_hidden_repr
        yes_minus_no_hidden_representations_list.append(yes_minus_no_hidden_repr)

    return torch.cat(yes_minus_no_hidden_representations_list)


def compute_seen_and_unseen_responses_yes_minus_no_hidden_representations(
        logger,
        model,
        tokenizer,
        datamodule,
        eval_output_tokens_matching_yes: List[str],
        eval_output_tokens_matching_no: List[str],
        eval_prefix_matching_yes: str,
        eval_prefix_matching_no: str,
        device,
        split: str = "train"):
    logger.info(f"Starting to compute hidden representations for seen responses in split: {split}")
    dataloader = datamodule.prepared_train_dataloader if split == "train" else datamodule.prepared_test_dataloader
    seen_yes_minus_no_hidden_representations = __compute_yes_minus_no_hidden_representations_for_seen_responses(dataloader, model, tokenizer, device)

    logger.info(f"Starting to compute hidden representations for unseen responses in split: {split}")
    dataset = datamodule.train_dataset if split == "train" else datamodule.test_dataset
    unseen_yes_minus_no_hidden_representations = __compute_yes_minus_no_hidden_representations_for_unseen_responses(
        dataset,
        model,
        tokenizer,
        eval_output_tokens_matching_yes,
        eval_output_tokens_matching_no,
        eval_prefix_matching_yes,
        eval_prefix_matching_no,
        batch_size=datamodule.batch_size,
        device=device
    )

    return seen_yes_minus_no_hidden_representations, unseen_yes_minus_no_hidden_representations


def __compute_and_log_metrics(logger,
                              train_seen_y_min_n_hidden_representations: torch.Tensor,
                              train_unseen_y_min_n_hidden_representations: torch.Tensor,
                              test_unseen_y_min_n_hidden_representations: torch.Tensor,
                              train_max_margin_classifier: torch.Tensor = None):
    train_seen_train_unseen_inner_products = torch.matmul(train_seen_y_min_n_hidden_representations,
                                                          train_unseen_y_min_n_hidden_representations.t())
    train_seen_test_unseen_inner_products = torch.matmul(train_seen_y_min_n_hidden_representations,
                                                         test_unseen_y_min_n_hidden_representations.t())

    train_seen_train_unseen_cosine_similarities = train_seen_train_unseen_inner_products / (
            train_seen_y_min_n_hidden_representations.norm(dim=1).unsqueeze(1)
            * train_unseen_y_min_n_hidden_representations.norm(dim=1).unsqueeze(0)
    )
    train_seen_test_unseen_cosine_similarities = train_seen_test_unseen_inner_products / (
            train_seen_y_min_n_hidden_representations.norm(dim=1).unsqueeze(1)
            * test_unseen_y_min_n_hidden_representations.norm(dim=1).unsqueeze(0)
    )

    train_seen_train_unseen_mean_inner_prod = train_seen_train_unseen_inner_products.mean().item()
    train_seen_test_unseen_mean_inner_prod = train_seen_test_unseen_inner_products.mean().item()

    train_seen_train_unseen_min_inner_prod = train_seen_train_unseen_inner_products.min().item()
    train_seen_test_unseen_min_inner_prod = train_seen_test_unseen_inner_products.min().item()

    train_seen_train_unseen_first_quartile_inner_prod = torch.quantile(train_seen_train_unseen_inner_products, q=0.25).item()
    train_seen_test_unseen_first_quartile_inner_prod = torch.quantile(train_seen_test_unseen_inner_products, q=0.25).item()

    train_seen_train_unseen_med_inner_prod = train_seen_train_unseen_inner_products.median().item()
    train_seen_test_unseen_med_inner_prod = train_seen_test_unseen_inner_products.median().item()

    train_seen_train_unseen_third_quartile_inner_prod = torch.quantile(train_seen_train_unseen_inner_products, q=0.75).item()
    train_seen_test_unseen_third_quartile_inner_prod = torch.quantile(train_seen_test_unseen_inner_products, q=0.75).item()

    train_seen_train_unseen_max_inner_prod = train_seen_train_unseen_inner_products.max().item()
    train_seen_test_unseen_max_inner_prod = train_seen_test_unseen_inner_products.max().item()

    train_seen_train_unseen_mean_cosine_sim = train_seen_train_unseen_cosine_similarities.mean().item()
    train_seen_test_unseen_mean_cosine_sim = train_seen_test_unseen_cosine_similarities.mean().item()

    train_seen_train_unseen_min_cosine_sim = train_seen_train_unseen_cosine_similarities.min().item()
    train_seen_test_unseen_min_cosine_sim = train_seen_test_unseen_cosine_similarities.min().item()

    train_seen_train_unseen_first_quartile_cosine_sim = torch.quantile(train_seen_train_unseen_cosine_similarities, q=0.25).item()
    train_seen_test_unseen_first_quartile_cosine_sim = torch.quantile(train_seen_test_unseen_cosine_similarities, q=0.25).item()

    train_seen_train_unseen_med_cosine_sim = train_seen_train_unseen_cosine_similarities.median().item()
    train_seen_test_unseen_med_cosine_sim = train_seen_test_unseen_cosine_similarities.median().item()

    train_seen_train_unseen_third_quartile_cosine_sim = torch.quantile(train_seen_train_unseen_cosine_similarities, q=0.75).item()
    train_seen_test_unseen_third_quartile_cosine_sim = torch.quantile(train_seen_test_unseen_cosine_similarities, q=0.75).item()

    train_seen_train_unseen_max_cosine_sim = train_seen_train_unseen_cosine_similarities.max().item()
    train_seen_test_unseen_max_cosine_sim = train_seen_test_unseen_cosine_similarities.max().item()

    train_seen_train_unseen_frac_pos_inner_prod = (train_seen_train_unseen_inner_products > 0).float().mean().item()
    train_seen_test_unseen_frac_pos_inner_prod = (train_seen_test_unseen_inner_products > 0).float().mean().item()

    logger.info(
        "\n------------------------------------------------------------------------------------------------------------------------------\n"
        "Hidden representation statistics:\n"
        "\n[Train Seen vs Train Unseen]\n"
        "  --- Inner Product ---\n"
        f"    Mean: {train_seen_train_unseen_mean_inner_prod}\n"
        f"    Min: {train_seen_train_unseen_min_inner_prod}\n"
        f"    1st quartile: {train_seen_train_unseen_first_quartile_inner_prod}\n"
        f"    Median: {train_seen_train_unseen_med_inner_prod}\n"
        f"    3rd quartile: {train_seen_train_unseen_third_quartile_inner_prod}\n"
        f"    Max: {train_seen_train_unseen_max_inner_prod}\n"
        f"    Fraction positive: {train_seen_train_unseen_frac_pos_inner_prod}\n"
        "  --- Cosine Similarity ---\n"
        f"    Mean: {train_seen_train_unseen_mean_cosine_sim}\n"
        f"    Min: {train_seen_train_unseen_min_cosine_sim}\n"
        f"    1st quartile: {train_seen_train_unseen_first_quartile_cosine_sim}\n"
        f"    Median: {train_seen_train_unseen_med_cosine_sim}\n"
        f"    3rd quartile: {train_seen_train_unseen_third_quartile_cosine_sim}\n"
        f"    Max: {train_seen_train_unseen_max_cosine_sim}\n"
        "\n[Train Seen vs Test Unseen]\n"
        "  --- Inner Product ---\n"
        f"    Mean: {train_seen_test_unseen_mean_inner_prod}\n"
        f"    Min: {train_seen_test_unseen_min_inner_prod}\n"
        f"    1st quartile: {train_seen_test_unseen_first_quartile_inner_prod}\n"
        f"    Median: {train_seen_test_unseen_med_inner_prod}\n"
        f"    3rd quartile: {train_seen_test_unseen_third_quartile_inner_prod}\n"
        f"    Max: {train_seen_test_unseen_max_inner_prod}\n"
        f"    Fraction positive: {train_seen_test_unseen_frac_pos_inner_prod}\n"
        "  --- Cosine Similarity ---\n"
        f"    Mean: {train_seen_test_unseen_mean_cosine_sim}\n"
        f"    Min: {train_seen_test_unseen_min_cosine_sim}\n"
        f"    1st quartile: {train_seen_test_unseen_first_quartile_cosine_sim}\n"
        f"    Median: {train_seen_test_unseen_med_cosine_sim}\n"
        f"    3rd quartile: {train_seen_test_unseen_third_quartile_cosine_sim}\n"
        f"    Max: {train_seen_test_unseen_max_cosine_sim}\n"
        "------------------------------------------------------------------------------------------------------------------------------"
    )

    if train_max_margin_classifier is not None:
        classifier_and_train_seen_cosine_similarities = torch.matmul(
            train_max_margin_classifier,
            train_seen_y_min_n_hidden_representations.t()
        ) / (train_max_margin_classifier.norm() * train_seen_y_min_n_hidden_representations.norm(dim=1))

        classifier_and_train_unseen_cosine_similarities = torch.matmul(
            train_max_margin_classifier,
            train_unseen_y_min_n_hidden_representations.t()
        ) / (train_max_margin_classifier.norm() * train_unseen_y_min_n_hidden_representations.norm(dim=1))

        classifier_and_test_unseen_cosine_similarities = torch.matmul(
            train_max_margin_classifier,
            test_unseen_y_min_n_hidden_representations.t()
        ) / (train_max_margin_classifier.norm() * test_unseen_y_min_n_hidden_representations.norm(dim=1))

        classifier_and_train_seen_mean_cosine_sim = classifier_and_train_seen_cosine_similarities.mean().item()
        classifier_and_train_unseen_mean_cosine_sim = classifier_and_train_unseen_cosine_similarities.mean().item()
        classifier_and_test_unseen_mean_cosine_sim = classifier_and_test_unseen_cosine_similarities.mean().item()

        classifier_and_train_unseen_min_cosine_sim = classifier_and_train_unseen_cosine_similarities.min().item()
        classifier_and_test_unseen_min_cosine_sim = classifier_and_test_unseen_cosine_similarities.min().item()

        classifier_and_train_unseen_first_quartile_cosine_sim = torch.quantile(classifier_and_train_unseen_cosine_similarities, q=0.25).item()
        classifier_and_test_unseen_first_quartile_cosine_sim = torch.quantile(classifier_and_test_unseen_cosine_similarities, q=0.25).item()

        classifier_and_train_unseen_med_cosine_sim = classifier_and_train_unseen_cosine_similarities.median().item()
        classifier_and_test_unseen_med_cosine_sim = classifier_and_test_unseen_cosine_similarities.median().item()

        classifier_and_train_unseen_third_quartile_cosine_sim = torch.quantile(classifier_and_train_unseen_cosine_similarities, q=0.75).item()
        classifier_and_test_unseen_third_quartile_cosine_sim = torch.quantile(classifier_and_test_unseen_cosine_similarities, q=0.75).item()

        classifier_and_train_unseen_max_cosine_sim = classifier_and_train_unseen_cosine_similarities.max().item()
        classifier_and_test_unseen_max_cosine_sim = classifier_and_test_unseen_cosine_similarities.max().item()

        classifier_and_train_seen_frac_pos = (classifier_and_train_seen_cosine_similarities > 0).float().mean().item()
        classifier_train_unseen_frac_pos = (classifier_and_train_unseen_cosine_similarities > 0).float().mean().item()
        classifier_test_unseen_frac_pos = (classifier_and_test_unseen_cosine_similarities > 0).float().mean().item()

        logger.info(
            "\n------------------------------------------------------------------------------------------------------------------------------\n"
            "Max-margin classifier statistics:\n"
            "\n[Train classifier vs Train Seen]\n"
            "  --- Cosine Similarity ---\n"
            f"    Mean: {classifier_and_train_seen_mean_cosine_sim}\n"
            f"    Fraction positive: {classifier_and_train_seen_frac_pos}\n"
            "\n[Train classifier vs Train Unseen]\n"
            "  --- Cosine Similarity ---\n"
            f"    Mean: {classifier_and_train_unseen_mean_cosine_sim}\n"
            f"    Min: {classifier_and_train_unseen_min_cosine_sim}\n"
            f"    1st quartile: {classifier_and_train_unseen_first_quartile_cosine_sim}\n"
            f"    Median: {classifier_and_train_unseen_med_cosine_sim}\n"
            f"    3rd quartile: {classifier_and_train_unseen_third_quartile_cosine_sim}\n"
            f"    Max: {classifier_and_train_unseen_max_cosine_sim}\n"
            f"    Fraction positive: {classifier_train_unseen_frac_pos}\n"
            "\n[Train classifier vs Test Unseen]\n"
            "  --- Cosine Similarity ---\n"
            f"    Mean: {classifier_and_test_unseen_mean_cosine_sim}\n"
            f"    Min: {classifier_and_test_unseen_min_cosine_sim}\n"
            f"    1st quartile: {classifier_and_test_unseen_first_quartile_cosine_sim}\n"
            f"    Median: {classifier_and_test_unseen_med_cosine_sim}\n"
            f"    3rd quartile: {classifier_and_test_unseen_third_quartile_cosine_sim}\n"
            f"    Max: {classifier_and_test_unseen_max_cosine_sim}\n"
            f"    Fraction positive: {classifier_test_unseen_frac_pos}\n"
            "------------------------------------------------------------------------------------------------------------------------------"
        )


def __compute_binary_max_margin_linear_predictor(X: torch.Tensor, y: torch.Tensor, normalize: bool = False):
    """
    Computes the binary max-margin linear predictor (without bias) for linearly separable data.
    @param X: Tensor of input samples with shape (num_samples, dim).
    @param y: Tensor holding zeroes and ones with the binary labels of shape (num_samples,)
    @param normalize: If True, will normalize the max-margin predictor to unit norm
    @return: Max-margin linear predictor as tensor of shape (dim,) if found, and None otherwise (e.g. data is not linearly separable).
    """
    np_X = X.detach().cpu().numpy()
    np_y = y.detach().cpu().numpy()
    n_samples = np_X.shape[0]
    dim = np_X.shape[1]

    # Convert labels to -1 and 1
    np_y = np.where(np_y == 0, -1, np_y)

    # Construct the quadratic programming problem
    P = matrix(np.identity(dim, dtype=float))
    q = matrix(np.zeros((dim,), dtype=float))
    G = matrix(-np_X * np.expand_dims(np_y, axis=1).astype(np.double))
    h = matrix(-np.ones((n_samples,), dtype=float))

    # Solve the quadratic programming problem
    try:
        solution = solvers.qp(P, q, G, h)
        w = torch.tensor(np.array(solution['x']), dtype=torch.float, device=X.device).flatten()
        return w / torch.norm(w) if normalize else w
    except ValueError:
        return None


@torch.no_grad()
def main(config: dict):
    model_name = config["model"]
    dataset_path = config["dataset"]
    device = torch.device(
        f"cuda:{config['gpu_id']}"
        if torch.cuda.is_available() and config["gpu_id"] >= 0
        else "cpu"
    )

    dataset_display_name = dataset_path.split("/")[-1].split(".")[0]
    subdir_name = model_name.split("/")[-1] + "_" + dataset_display_name
    logger = logging_utils.create_logger(
        file_logging=not config["dont_save_logs"],
        log_dir=os.path.join(config["output_dir"], subdir_name),
        log_file_name_prefix=f"log_samples_{config['num_train_samples']}",
    )
    logger.info(f"Config: {config}")

    try:
        start_time = datetime.utcnow()

        logger.info(f"======================================================================================================")
        logger.info(f"Model: '{model_name}', Dataset: '{dataset_path}'")
        logger.info(f"======================================================================================================\n")

        tokenizer, model = __load_tokenizer_and_model(
            logger, model_name, cache_dir=config["cache_dir"], device=device
        )

        datamodule = PersonaControlledDataModule(
            path=dataset_path,
            tokenizer=tokenizer,
            num_train_samples=config["num_train_samples"],
            num_test_samples=config["num_test_samples"],
            answer_matching_behavior_to_use=config["answer_matching_behavior_to_use"],
            output_tokens_matching_yes=config["output_tokens_matching_yes"],
            output_tokens_matching_no=config["output_tokens_matching_no"],
            prefixes_matching_yes=config["prefixes_matching_yes"],
            prefixes_matching_no=config["prefixes_matching_no"],
            batch_size=config["batch_size"],
            random_seed=config["dataset_prep_random_seed"]
        )
        datamodule.setup()

        train_seen_y_min_n_hidden_representations, train_unseen_y_min_n_hidden_representations = (
            compute_seen_and_unseen_responses_yes_minus_no_hidden_representations(
                logger=logger,
                model=model,
                tokenizer=tokenizer,
                datamodule=datamodule,
                device=device,
                eval_output_tokens_matching_yes=config["eval_output_tokens_matching_yes"],
                eval_output_tokens_matching_no=config["eval_output_tokens_matching_no"],
                eval_prefix_matching_yes=config["eval_prefix_matching_yes"],
                eval_prefix_matching_no=config["eval_prefix_matching_no"],
                split="train"
            )
        )

        _, test_unseen_y_min_n_hidden_representations = (
            compute_seen_and_unseen_responses_yes_minus_no_hidden_representations(
                logger=logger,
                model=model,
                tokenizer=tokenizer,
                datamodule=datamodule,
                device=device,
                eval_output_tokens_matching_yes=config["eval_output_tokens_matching_yes"],
                eval_output_tokens_matching_no=config["eval_output_tokens_matching_no"],
                eval_prefix_matching_yes=config["eval_prefix_matching_yes"],
                eval_prefix_matching_no=config["eval_prefix_matching_no"],
                split="test"
            )
        )

        train_max_margin_classifier = __compute_binary_max_margin_linear_predictor(train_seen_y_min_n_hidden_representations,
                                                                                   torch.ones(train_seen_y_min_n_hidden_representations.shape[0],
                                                                                              device=device))
        if train_max_margin_classifier is None:
            logger.info("Train yes minus no hidden representations not linearly separable.")

        __compute_and_log_metrics(logger,
                                  train_seen_y_min_n_hidden_representations,
                                  train_unseen_y_min_n_hidden_representations,
                                  test_unseen_y_min_n_hidden_representations,
                                  train_max_margin_classifier)

        hidden_repr_sim_results = {
            "train_seen_y_min_n_hidden_representations": train_seen_y_min_n_hidden_representations,
            "train_unseen_y_min_n_hidden_representations": train_unseen_y_min_n_hidden_representations,
            "test_unseen_y_min_n_hidden_representations": test_unseen_y_min_n_hidden_representations
        }

        torch.save(hidden_repr_sim_results, os.path.join(config["output_dir"], subdir_name, f"hidden_repr_similarity_results.pt"))

        end_time = datetime.utcnow()
        logger.info(f"Finished script, time took: {end_time - start_time}")
    except Exception:
        logger.exception("Exception while running script.")
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="outputs/persona_hidden_repr", help="Directory to save log file to", )
    p.add_argument("--dataset", type=str, default="data_files/persona/agreeableness.jsonl", help="Dataset to use")
    p.add_argument("--cache_dir", type=str, default=None, help="Directory of cache for HuggingFace models and datasets")
    p.add_argument("--dont_save_logs", action="store_true", help="Only log to console, and not to a file")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model to use")
    p.add_argument("--gpu_id", type=int, default=0, help="GPU id to use (-1 for CPU)")

    p.add_argument("--num_train_samples", type=int, default=80, help="Number of training samples to use")
    p.add_argument("--num_test_samples", type=int, default=40, help="Number of validation samples to use")
    p.add_argument("--dataset_prep_random_seed", type=int, default=875, help="Random seed for preparing the data")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--answer_matching_behavior_to_use", type=str, default="Yes",
                   help="If not empty, will use only training samples with the given 'answer_matching_behavior' value. Should be "
                        "either 'Yes' or 'No'.")
    p.add_argument("--output_tokens_matching_yes", nargs="+", type=str, default=["yes", "sure"], help="Tokens corresponding to original answer 'Yes'")
    p.add_argument("--output_tokens_matching_no", nargs="+", type=str, default=["no", "never"], help="Tokens corresponding to original answer 'No'")
    p.add_argument("--prefixes_matching_yes", nargs="+", type=str, default=["In my view, the answer is ", "A truthful reply is "],
                   help="Prefixes before answers matching the original answer 'Yes'")
    p.add_argument("--prefixes_matching_no", nargs="+", type=str, default=["My judgement: ", "Considering the statement, I say "],
                   help="Prefixes before answers matching the original answer 'No'")
    p.add_argument("--eval_output_tokens_matching_yes", nargs="+", type=str, default=["certainly", "absolutely"],
                   help="Tokens corresponding to original answer 'Yes' to be used in evaluation")
    p.add_argument("--eval_output_tokens_matching_no", nargs="+", type=str, default=["not really", "nope"],
                   help="Tokens corresponding to original answer 'No' to be used in evaluation")
    p.add_argument("--eval_prefix_matching_yes", type=str, default="My response would be ",
                   help="Prefix corresponding to original answer 'Yes' to be used in evaluation")
    p.add_argument("--eval_prefix_matching_no", nargs="+", default="I lean toward ",
                   help="Prefix corresponding to original answer 'No' to be used in evaluation")

    args = p.parse_args()
    main(args.__dict__)
