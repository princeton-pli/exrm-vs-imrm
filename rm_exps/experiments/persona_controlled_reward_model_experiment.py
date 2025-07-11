import logging
import os
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl.trainer.utils import prepare_deepspeed

from common.data.modules import DataModule
from common.evaluation.evaluators import TrainEvaluator, Evaluator, TrainBatchOutputEvaluator
from common.experiment import FitExperimentBase, ExperimentResult
from common.experiment.fit_experiment_base import ScoreInfo
from common.train.callbacks import Callback
from common.train.fit_output import FitOutput
from common.train.trainer import Trainer
from rm_exps.data.persona_controlled_datamodule import PersonaControlledDataModule
from rm_exps.eval.persona_reward_model_evaluator import PersonaRewardModelEvaluator
from rm_exps.eval.sft_evaluator import SFTEvaluator
from rm_exps.train.reward_model_trainer import RewardModelTrainer
from utils.sharedmisc import update_tokenizer, update_model_num_embeddings_and_special_tokens


class PersonaControlledRewardModelExperiment(FitExperimentBase):

    @staticmethod
    def add_experiment_specific_args(parser):
        FitExperimentBase.add_experiment_base_specific_args(parser)

        parser.add_argument("--dataset_path", type=str, default="data_files/persona/agreeableness.jsonl", help="Path to the Persona dataset to use.")
        parser.add_argument("--num_train_samples", type=int, default=400, help="Number of training samples to use")
        parser.add_argument("--num_test_samples", type=int, default=50, help="Number of validation samples to use")
        parser.add_argument("--dataset_prep_random_seed", type=int, default=-1, help="Random seed for preparing the data")
        parser.add_argument("--batch_size", type=int, default=2)

        parser.add_argument("--answer_matching_behavior_to_use", type=str, default="Yes",
                            help="If not empty, will use only training samples with the "
                                 "given 'answer_matching_behavior' value. Should be "
                                 "either 'Yes' or 'No'.")
        parser.add_argument("--output_tokens_matching_yes", nargs="+", type=str, default=["yes"],
                            help="Tokens corresponding to original answer 'Yes'")
        parser.add_argument("--output_tokens_matching_no", nargs="+", type=str, default=["no"],
                            help="Tokens corresponding to original answer 'No'")
        parser.add_argument("--prefixes_matching_yes", nargs="+", type=str, default=["In my view, the answer is ",
                                                                                     "A truthful reply is "],
                            help="Prefixes before answers matching the original answer 'Yes'")
        parser.add_argument("--prefixes_matching_no", nargs="+", type=str, default=["My judgement: ",
                                                                                    "Considering the statement, I say "],
                            help="Prefixes before answers matching the original answer 'No'")
        parser.add_argument("--eval_output_tokens_matching_yes", nargs="+", type=str, default=["Yes", "Sure", "sure", "Absolutely", "Certainly"],
                            help="Tokens corresponding to original answer 'Yes' to be used in evaluation")
        parser.add_argument("--eval_output_tokens_matching_no", nargs="+", type=str, default=["No", "Never", "never", "Nope"],
                            help="Tokens corresponding to original answer 'No' to be used in evaluation")
        parser.add_argument("--eval_prefixes_matching_yes", nargs="+", type=str, default=["My response would be "],
                            help="Prefixes corresponding to original answer 'Yes' to be used in evaluation")
        parser.add_argument("--eval_prefixes_matching_no", nargs="+", type=str, default=["I lean toward "],
                            help="Prefixes corresponding to original answer 'No' to be used in evaluation")

        parser.add_argument("--model", type=str, default="EleutherAI/pythia-1b", help="Model to use")
        parser.add_argument("--model_cache_dir", type=str, default=None, help="Hugging Face cache dir.")
        parser.add_argument("--model_parallel_without_accelerate", action="store_true",
                            help="Whether to use model parallelism (only use when running script without accelerate)")
        parser.add_argument("--kl_coeff", type=float, default=0.01, help="KL divergence coefficient for IM-RM")
        parser.add_argument("--objective", type=str, default="ex_rm", help="Objective type to use. Supports 'ex_rm', 'im_rm', 'gen_rm', and 'sft'.")
        parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use. Supports 'adam', 'rmsprop', and 'sgd'.")
        parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")

        parser.add_argument("--save_model", action="store_true", help="Save the model at the end of the experiment")

    def initialize(self, config: dict, state: dict):
        super().initialize(config, state)

        logger = state["logger"]
        tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True, cache_dir=config["model_cache_dir"])
        tokenizer, num_added_tokens = update_tokenizer(tokenizer, logger=logger)

        device_map = "auto" if config["model_parallel_without_accelerate"] else None
        ref_model = None
        if config["objective"] == "ex_rm":
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=config["model"],
                num_labels=1,
                device_map=device_map,
                trust_remote_code=True,
                cache_dir=config["model_cache_dir"]
            )


        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=config["model"],
                device_map=device_map,
                trust_remote_code=True,
                cache_dir=config["model_cache_dir"]
            )
            if config["objective"] == "im_rm":
                ref_model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=config["model"],
                    device_map=device_map,
                    trust_remote_code=True,
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
        datamodule = PersonaControlledDataModule(path=config["dataset_path"], tokenizer=state["tokenizer"],
                                                 num_train_samples=config["num_train_samples"],
                                                 num_test_samples=config["num_test_samples"],
                                                 answer_matching_behavior_to_use=config["answer_matching_behavior_to_use"],
                                                 output_tokens_matching_yes=config["output_tokens_matching_yes"],
                                                 output_tokens_matching_no=config["output_tokens_matching_no"],
                                                 prefixes_matching_yes=config["prefixes_matching_yes"],
                                                 prefixes_matching_no=config["prefixes_matching_no"],
                                                 batch_size=config["batch_size"],
                                                 random_seed=config["dataset_prep_random_seed"])
        datamodule.setup()
        return datamodule

    def create_model(self, datamodule: DataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        model = state["model"]
        accelerator = state["accelerator"]

        if config["optimizer"] == "adam":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0)
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

    def create_train_and_validation_evaluators(self, model: nn.Module, datamodule: DataModule, device, config: dict, state: dict,
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
            val_evaluator = PersonaRewardModelEvaluator(state["accelerator"],
                                                        model,
                                                        objective=config["objective"],
                                                        tokenizer=state["tokenizer"],
                                                        datamodule=datamodule,
                                                        ref_model=state["ref_model"],
                                                        kl_coeff=config["kl_coeff"],
                                                        output_tokens_matching_yes=config["output_tokens_matching_yes"],
                                                        output_tokens_matching_no=config["output_tokens_matching_no"],
                                                        prefixes_matching_yes=config["prefixes_matching_yes"],
                                                        prefixes_matching_no=config["prefixes_matching_no"],
                                                        eval_output_tokens_matching_yes=config["eval_output_tokens_matching_yes"],
                                                        eval_output_tokens_matching_no=config["eval_output_tokens_matching_no"],
                                                        eval_prefixes_matching_yes=config["eval_prefixes_matching_yes"],
                                                        eval_prefixes_matching_no=config["eval_prefixes_matching_no"])

        return train_evaluator, val_evaluator

    def create_additional_metadata_to_log(self, model: torch.nn.Module, datamodule: PersonaControlledDataModule,
                                          config: dict, state: dict, logger: logging.Logger) -> dict:
        additional_metadata = super().create_additional_metadata_to_log(model, datamodule, config, state, logger)
        additional_metadata["num train samples"] = len(datamodule.train_dataset)
        additional_metadata["num val samples"] = len(datamodule.test_dataset)
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
                                  train_evaluator=train_evaluator, val_evaluator=val_evaluator, callback=callback)

    def on_experiment_end(self, model: nn.Module, datamodule: PersonaControlledDataModule, trainer: Trainer, fit_output: FitOutput,
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

        validation_metrics = {}
        tracked_value = list(fit_output.val_tracked_values.values())[0]
        validation_metrics["epochs_with_values"] = tracked_value.epochs_with_values
        for metric_name, tracked_value in fit_output.val_tracked_values.items():
            validation_metrics[metric_name] = tracked_value.epoch_values

        if state["accelerator"].is_main_process:
            torch.save(validation_metrics, os.path.join(state["experiment_dir"], "eval_metrics.pt"))
