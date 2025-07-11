import logging
import os
from typing import Tuple

import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl.trainer.utils import prepare_deepspeed

from common.data.modules import DataModule
from common.evaluation.evaluators import TrainEvaluator, Evaluator, TrainBatchOutputEvaluator, ComposeEvaluator
from common.experiment import FitExperimentBase, ExperimentResult
from common.experiment.fit_experiment_base import ScoreInfo
from common.train.callbacks import Callback
from common.train.fit_output import FitOutput
from common.train.trainer import Trainer
from rm_exps.data.preference_datamodule import PreferenceDataModule
from rm_exps.eval.ham_cycle_gen_evaluator import HamiltonianCycleGenerationEvaluator
from rm_exps.eval.reward_model_evaluator import RewardModelEvaluator
from rm_exps.eval.sft_evaluator import SFTEvaluator
from rm_exps.train.reward_model_trainer import RewardModelTrainer
from utils.data_utils import HamiltonianCycleDatasetLoader
from utils.sharedmisc import update_tokenizer, update_model_num_embeddings_and_special_tokens


class RewardModelExperiment(FitExperimentBase):

    @staticmethod
    def add_experiment_specific_args(parser):
        FitExperimentBase.add_experiment_base_specific_args(parser)

        parser.add_argument("--dataset_path", type=str, required=True,
                            help="Path to the preference dataset to use. Assumes that the dataset has a 'train' and 'test' splits, where each example "
                                 "in the dataset has the following fields: 'prompt', 'chosen', and 'rejected'.")
        parser.add_argument("--num_train_samples", type=int, default=-1, help="Number of training samples to use")
        parser.add_argument("--num_test_samples", type=int, default=-1, help="Number of validation samples to use")
        parser.add_argument("--dataset_prep_random_seed", type=int, default=-1, help="Random seed for preparing the data")
        parser.add_argument("--batch_size", type=int, default=8)

        parser.add_argument("--evaluation_datasets_paths", type=str, nargs="+", default=[],
                            help="Paths to the evaluation datasets to use.")
        parser.add_argument("--evaluation_datasets_names", type=str, nargs="+", default=[],
                            help="Names of the evaluation datasets to use. Used for easier identification and printing."
                                 "This argument must have the same number of entries as 'evaluation_datasets_paths'.")
        parser.add_argument("--evalution_datasets_splits", type=str, nargs="+", default=[],
                            help="Split name per evaluation datasets. "
                                 "This argument must have the same number of entries as 'evaluation_datasets_paths'.")

        parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model to use.")
        parser.add_argument("--model_cache_dir", type=str, default=None, help="Hugging Face cache dir.")
        parser.add_argument("--use_bf16", action="store_true", help="Whether to use bf16 precision.")
        parser.add_argument("--model_parallel_without_accelerate", action="store_true",
                            help="Whether to use model parallelism (only use when running script without accelerate)")
        parser.add_argument("--kl_coeff", type=float, default=0.01, help="KL divergence coefficient for IM-RM")
        parser.add_argument("--objective", type=str, default="ex_rm", help="Objective type to use. Supports 'ex_rm', 'im_rm', 'gen_rm', and 'sft'.")
        parser.add_argument("--use_all_response_hidden_embeddings", action="store_true",
                            help="If objective is 'ex_rm', use linear head over the average of the mean response hidden embeddings as opposed "
                                 "to just the final embedding.")
        parser.add_argument("--no_ref_model", action="store_true", help="If objective is 'im_rm', do not use a reference model.")
        parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use. Supports 'adam', 'rmsprop', and 'sgd'.")
        parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
        parser.add_argument("--save_model", action="store_true", help="Save the model at the end of the experiment")

        parser.add_argument("--is_ham_cycle_task", action="store_true",
                            help="Add special tokens for verifying Hamiltonian cycle task. Should only be set to True when running experiments on "
                                 "the Hamiltonian cycle task.")
        parser.add_argument("--ham_cycle_gen_num_tries", type=int, default=1,
                            help="Number of tries to generate a Hamiltonian cycle during evaluation. Only relevant if is_ham_cycle_task is True and "
                                 "objective is 'im_rm'.")

    def initialize(self, config: dict, state: dict):
        super().initialize(config, state)

        logger = state["logger"]
        tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True, cache_dir=config["model_cache_dir"])
        tokenizer, num_added_tokens = update_tokenizer(tokenizer, logger=logger)

        if config.get("is_ham_cycle_task", False):
            num_added_tokens += tokenizer.add_tokens([HamiltonianCycleDatasetLoader.SEP_TOKEN, HamiltonianCycleDatasetLoader.EDGE_SEP_TOKEN])

        device_map = "auto" if config["model_parallel_without_accelerate"] else None
        ref_model = None
        if config["objective"] == "ex_rm":
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=config["model"],
                num_labels=1,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if config["use_bf16"] else None,
                cache_dir=config["model_cache_dir"]
            )



        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=config["model"],
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if config["use_bf16"] else None,
                cache_dir=config["model_cache_dir"]
            )
            if config["objective"] == "im_rm" and not config["no_ref_model"]:
                ref_model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=config["model"],
                    device_map=device_map,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if config["use_bf16"] else None,
                    cache_dir=config["model_cache_dir"]
                )
                ref_model.eval()
                for param in ref_model.parameters():
                    param.requires_grad = False

        if num_added_tokens > 0:
            logger.warning("Updating model embeddings since tokens were added to the tokenizer.")
            model = update_model_num_embeddings_and_special_tokens(model, tokenizer)
            if ref_model is not None:
                ref_model = update_model_num_embeddings_and_special_tokens(ref_model, tokenizer)

        state["tokenizer"] = tokenizer
        state["model"] = model
        state["ref_model"] = ref_model

    def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> DataModule:
        datamodule = PreferenceDataModule(path=config["dataset_path"], tokenizer=state["tokenizer"],
                                          num_train_samples=config["num_train_samples"],
                                          num_test_samples=config["num_test_samples"],
                                          batch_size=config["batch_size"],
                                          random_seed=config["dataset_prep_random_seed"])
        datamodule.setup()
        return datamodule

    def create_model(self, datamodule: PreferenceDataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        model = state["model"]
        accelerator = state["accelerator"]

        if config["optimizer"] == "adam":
            optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0, lr=config["lr"])
        elif config["optimizer"] == "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=config["lr"])
        elif config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
        else:
            raise ValueError(f"Optimizer '{config['optimizer']}' is not supported.")

        model, optimizer, datamodule.prepared_train_dataloader = accelerator.prepare(model, optimizer, datamodule.prepared_train_dataloader)
        state["model"] = model
        state["optimizer"] = optimizer
        datamodule.prepared_test_dataloader = accelerator.prepare(datamodule.prepared_test_dataloader)

        if state["ref_model"] is not None:
            if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
                state["ref_model"] = prepare_deepspeed(state["ref_model"], config["batch_size"])
            else:
                state["ref_model"] = state["ref_model"].to(accelerator.device)

        return model

    def create_train_and_validation_evaluators(self, model: nn.Module, datamodule: PreferenceDataModule, device, config: dict, state: dict,
                                               logger: logging.Logger) -> Tuple[TrainEvaluator, Evaluator]:
        if config["objective"] == "sft":
            batch_output_metrics_to_track = ["train loss", "train chosen logprobs", "train rejected logprobs"]
            batch_output_metric_tags = ["train loss", "train logprobs", "train logprobs"]
        elif config["objective"] in ["ex_rm", "im_rm", "gen_rm"]:
            batch_output_metrics_to_track = ["train loss", "train chosen rewards", "train rejected rewards", "train reward margin", "train accuracy"]
            batch_output_metric_tags = ["train loss", "train rewards", "train rewards", "train reward margin", "train accuracy"]
        else:
            raise ValueError(f"Unknown objective: {config['objective']}")

        train_evaluator = TrainBatchOutputEvaluator(metric_names=batch_output_metrics_to_track, metric_tags=batch_output_metric_tags)

        if config["objective"] == "sft":
            val_evaluator = SFTEvaluator(state["accelerator"],
                                         model,
                                         state["tokenizer"],
                                         datamodule=datamodule)
        else:
            random_seed = config["dataset_prep_random_seed"] if config["dataset_prep_random_seed"] > 0 else -1
            dataloaders_dict = self.__create_test_dataloaders_dict(state["accelerator"], datamodule, config, config["num_test_samples"],
                                                                   batch_size=config["batch_size"], random_seed=random_seed)

            val_evaluator = RewardModelEvaluator(state["accelerator"],
                                                 model,
                                                 tokenizer=state["tokenizer"],
                                                 objective=config["objective"],
                                                 dataloaders_dict=dataloaders_dict,
                                                 ref_model=state["ref_model"],
                                                 kl_coeff=config["kl_coeff"],
                                                 use_all_response_hidden_embeddings=config["use_all_response_hidden_embeddings"],
                                                 no_ref_model=config["no_ref_model"])

        if config.get("is_ham_cycle_task", False) and config["objective"] == "im_rm":
            val_evaluator = ComposeEvaluator([val_evaluator, HamiltonianCycleGenerationEvaluator(accelerator=state["accelerator"],
                                                                                                 model=model,
                                                                                                 tokenizer=state["tokenizer"],
                                                                                                 datamodule=datamodule,
                                                                                                 num_tries=config.get("ham_cycle_gen_num_tries", 1),
                                                                                                 logger=logger)])

        return train_evaluator, val_evaluator

    def __create_test_dataloaders_dict(self, accelerator, datamodule: PreferenceDataModule, config: dict, num_test_samples: int, batch_size: int,
                                       random_seed: int) -> dict:
        dataset_paths = config["evaluation_datasets_paths"]
        dataset_names = config["evaluation_datasets_names"]
        splits = config["evalution_datasets_splits"]

        if len(dataset_paths) != len(splits) or len(dataset_paths) != len(dataset_names):
            raise ValueError(f"Number of evaluation dataset paths {(len(dataset_paths))} needs to match the number of split names {(len(splits))} "
                             f"and number of dataset names {len(dataset_names)}.")

        generator = torch.Generator().manual_seed(random_seed) if random_seed > 0 else None
        if num_test_samples > 0 and num_test_samples < len(datamodule.train_dataset):
            train_perm = torch.randperm(len(datamodule.train_dataset), generator=generator)
            train_sample_indices = torch.sort(train_perm[:num_test_samples]).values
            train_dataset = datamodule.train_dataset.select(train_sample_indices)

            batch_size = batch_size if batch_size > 0 else len(train_dataset)
            dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            train_dataloader = accelerator.prepare(dataloader)
        else:
            train_dataloader = datamodule.train_dataloader()

        dataloaders_dict = {
            "train": train_dataloader,
            "test": datamodule.val_dataloader()
        }

        for path, name, split in zip(dataset_paths, dataset_names, splits):
            dataset = load_from_disk(path)[split]
            num_test_samples = num_test_samples if num_test_samples > 0 else len(dataset)

            if num_test_samples < len(dataset):
                perm = torch.randperm(len(dataset), generator=generator)
                dataset = dataset.select(perm[:num_test_samples])

            batch_size = batch_size if batch_size > 0 else len(dataset)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            dataloaders_dict[f"{name}_{split}"] = accelerator.prepare(dataloader)

        return dataloaders_dict

    def create_additional_metadata_to_log(self, model: torch.nn.Module, datamodule: PreferenceDataModule,
                                          config: dict, state: dict, logger: logging.Logger) -> dict:
        additional_metadata = super().create_additional_metadata_to_log(model, datamodule, config, state, logger)
        additional_metadata["num train samples"] = len(datamodule.train_dataset)
        additional_metadata["num test samples"] = len(datamodule.test_dataset)
        train_sample = datamodule.train_dataset[0]
        additional_metadata["train_sample"] = {
            "prompt": train_sample["prompt"],
            "chosen": train_sample["chosen"],
            "rejected": train_sample["rejected"],
        }
        return additional_metadata

    def get_default_score_info(self, config: dict, state: dict) -> ScoreInfo:
        return ScoreInfo(metric_name="train loss", is_train_metric=True, largest=False, return_best_score=False)

    def create_trainer(self, model: nn.Module, datamodule: DataModule, train_evaluator: TrainEvaluator, val_evaluator: Evaluator,
                       callback: Callback, device, config: dict, state: dict, logger: logging.Logger):
        return RewardModelTrainer(accelerator=state["accelerator"], model=model, ref_model=state["ref_model"], tokenizer=state["tokenizer"],
                                  optimizer=state["optimizer"], kl_coeff=config["kl_coeff"], objective=config["objective"],
                                  use_all_response_hidden_embeddings=config["use_all_response_hidden_embeddings"],
                                  no_ref_model=config["no_ref_model"],
                                  train_evaluator=train_evaluator, val_evaluator=val_evaluator, callback=callback)

    def on_experiment_end(self, model: nn.Module, datamodule: PreferenceDataModule, trainer: Trainer, fit_output: FitOutput,
                          experiment_result: ExperimentResult, config: dict, state: dict, logger: logging.Logger):
        super().on_experiment_end(model, datamodule, trainer, fit_output, experiment_result, config, state, logger)

        accelerator = state["accelerator"]
        if accelerator.is_main_process and config["save_model"]:
            experiment_dir = state["experiment_dir"]
            model_dir_path = os.path.join(experiment_dir, f"model_epoch_{trainer.epoch}")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(model_dir_path)

        experiment_result.summary["num_train_samples"] = len(datamodule.train_dataset)
        experiment_result.summary["num_test_samples"] = len(datamodule.test_dataset)

        eval_metrics = {}
        tracked_value = list(fit_output.val_tracked_values.values())[0]
        eval_metrics["epochs_with_values"] = tracked_value.epochs_with_values
        for metric_name, tracked_value in fit_output.val_tracked_values.items():
            eval_metrics[metric_name] = tracked_value.epoch_values

        if state["accelerator"].is_main_process:
            torch.save(eval_metrics, os.path.join(state["experiment_dir"], "eval_metrics.pt"))
