import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator

from common.evaluation import metrics as metrics
from common.evaluation.evaluators import Evaluator, MetricsEvaluator
from utils.rm_utils import tokenize_and_prepare_concatenated_preference_batch, compute_rewards_with_explicit_rm, compute_rewards_with_implicit_rm, \
    convert_preference_batch_to_chat_format, compute_rewards_with_gen_rm, convert_preference_batch_to_gen_rm_format, \
    tokenize_and_prepare_concatenated_gen_rm_batch


class RewardModelEvaluator(Evaluator):
    LOSS_METRIC_NAME_TEMPLATE = "eval {dataset_name} loss"
    NORMALIZED_ABS_REWARD_MARGIN_METRIC_NAME_TEMPLATE = "eval {dataset_name} normalized abs reward margin"
    ACCURACY_METRIC_NAME_TEMPLATE = "eval {dataset_name} accuracy"
    CHOSEN_REWARD_METRIC_NAME_TEMPLATE = "eval {dataset_name} chosen reward"
    REJECTED_REWARD_METRIC_NAME_TEMPLATE = "eval {dataset_name} rejected reward"

    def __init__(self, accelerator: Accelerator, model: nn.Module, tokenizer, objective: str, dataloaders_dict: dict,
                 ref_model: nn.Module = None, kl_coeff: float = 0.05, use_all_response_hidden_embeddings: bool = False, no_ref_model: bool = False):
        self.accelerator = accelerator
        self.model = model
        self.ref_model = ref_model
        self.kl_coeff = kl_coeff
        self.tokenizer = tokenizer
        self.objective = objective
        self.dataloaders_dict = dataloaders_dict
        self.use_all_response_hidden_embeddings = use_all_response_hidden_embeddings
        self.no_ref_model = no_ref_model

        self.metric_infos = self.__create_metric_infos()
        self.metrics = {name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def __create_metric_infos(self):
        metric_infos = {}
        for name, dataloader in self.dataloaders_dict.items():
            loss_metric_name = self.LOSS_METRIC_NAME_TEMPLATE.format(dataset_name=name)
            normalized_abs_reward_margin_metric_name = self.NORMALIZED_ABS_REWARD_MARGIN_METRIC_NAME_TEMPLATE.format(dataset_name=name)
            acc_metric_name = self.ACCURACY_METRIC_NAME_TEMPLATE.format(dataset_name=name)
            chosen_reward_metric_name = self.CHOSEN_REWARD_METRIC_NAME_TEMPLATE.format(dataset_name=name)
            rejected_reward_metric_name = self.REJECTED_REWARD_METRIC_NAME_TEMPLATE.format(dataset_name=name)

            metric_infos.update({
                loss_metric_name: metrics.MetricInfo(loss_metric_name,
                                                     metrics.DummyAveragedMetric(),
                                                     tag="eval loss"),
                normalized_abs_reward_margin_metric_name: metrics.MetricInfo(normalized_abs_reward_margin_metric_name,
                                                                             metrics.DummyAveragedMetric(),
                                                                             tag="eval normalized abs reward margin"),
                acc_metric_name: metrics.MetricInfo(acc_metric_name,
                                                    metrics.DummyAveragedMetric(),
                                                    tag="eval accuracy"),
                chosen_reward_metric_name: metrics.MetricInfo(chosen_reward_metric_name,
                                                              metrics.DummyAveragedMetric(),
                                                              tag="eval reward"),
                rejected_reward_metric_name: metrics.MetricInfo(rejected_reward_metric_name,
                                                                metrics.DummyAveragedMetric(),
                                                                tag="eval reward")
            })

        return metric_infos

    def get_metric_infos(self):
        return self.metric_infos

    def get_metrics(self):
        return self.metrics

    def get_tracked_values(self):
        return self.tracked_values

    def __populate_metric_value(self, metric_name, value):
        metric = self.metrics[metric_name]
        metric(value)
        metric_tracked_value = self.tracked_values[metric_name]
        metric_tracked_value.add_batch_value(value)

    def __get_compute_rewards_fn(self):
        if self.objective == "ex_rm":
            return compute_rewards_with_explicit_rm
        elif self.objective == "im_rm":
            return compute_rewards_with_implicit_rm
        elif self.objective == "gen_rm":
            return compute_rewards_with_gen_rm
        else:
            raise ValueError(f"Unknown objective: {self.objective}. Supported objectives are 'ex_rm', 'im_rm', and 'gen_rm'.")

    def __compute_chosen_and_rejected_rewards(self, batch, compute_rewards_fn):
        if self.objective != "gen_rm":
            batch = convert_preference_batch_to_chat_format(self.tokenizer, batch)
            prompts = batch["prompt"]
            chosen_responses = batch["chosen"]
            rejected_responses = batch["rejected"]

            input_ids, attention_mask, loss_mask = tokenize_and_prepare_concatenated_preference_batch(self.tokenizer, prompts,
                                                                                                      chosen_responses,
                                                                                                      rejected_responses, self.accelerator.device)

            kwargs = {"use_all_response_hidden_embeddings": self.use_all_response_hidden_embeddings} if self.objective == "ex_rm" \
                else {"ref_model": self.ref_model, "kl_coeff": self.kl_coeff, "no_ref_model": self.no_ref_model}
            all_rewards = compute_rewards_fn(self.model, input_ids, attention_mask, loss_mask, **kwargs)
        else:
            batch = convert_preference_batch_to_gen_rm_format(self.tokenizer, batch)
            prompt_chosen = batch["prompt_chosen"]
            prompt_rejected = batch["prompt_rejected"]

            input_ids, attention_mask = tokenize_and_prepare_concatenated_gen_rm_batch(self.tokenizer, prompt_chosen,
                                                                                       prompt_rejected, self.accelerator.device)
            all_rewards = compute_rewards_fn(self.model, input_ids, attention_mask, self.tokenizer)

        chosen_reward = all_rewards[:input_ids.shape[0] // 2].detach()
        rejected_reward = all_rewards[input_ids.shape[0] // 2:].detach()
        return chosen_reward, rejected_reward

    def __compute_metrics(self):
        for name, dataloader in self.dataloaders_dict.items():
            compute_rewards_fn = self.__get_compute_rewards_fn()
            chosen_rewards_list = []
            rejected_rewards_list = []
            for batch in dataloader:
                chosen_reward, rejected_reward = self.__compute_chosen_and_rejected_rewards(batch, compute_rewards_fn)
                gathered_chosen_rewards, gathered_rejected_rewards = self.accelerator.gather_for_metrics((chosen_reward, rejected_reward))
                chosen_rewards_list.append(gathered_chosen_rewards)
                rejected_rewards_list.append(gathered_rejected_rewards)

            chosen_reward = torch.cat(chosen_rewards_list)
            rejected_reward = torch.cat(rejected_rewards_list)

            loss = - F.logsigmoid(chosen_reward - rejected_reward).mean().item()
            reward_std = torch.cat([chosen_reward, rejected_reward]).std()
            normalized_abs_reward_margin = ((chosen_reward - rejected_reward).abs() / (reward_std + 1e-8)).mean().item()
            accuracy = (chosen_reward > rejected_reward).float().mean().item() + (chosen_reward == rejected_reward).float().mean().item() / 2
            chosen_reward_mean = chosen_reward.mean().item()
            rejected_reward_mean = rejected_reward.mean().item()

            loss_metric_name = self.LOSS_METRIC_NAME_TEMPLATE.format(dataset_name=name)
            normalized_abs_reward_margin_metric_name = self.NORMALIZED_ABS_REWARD_MARGIN_METRIC_NAME_TEMPLATE.format(dataset_name=name)
            acc_metric_name = self.ACCURACY_METRIC_NAME_TEMPLATE.format(dataset_name=name)
            chosen_reward_metric_name = self.CHOSEN_REWARD_METRIC_NAME_TEMPLATE.format(dataset_name=name)
            rejected_reward_metric_name = self.REJECTED_REWARD_METRIC_NAME_TEMPLATE.format(dataset_name=name)

            self.__populate_metric_value(loss_metric_name, loss)
            self.__populate_metric_value(normalized_abs_reward_margin_metric_name, normalized_abs_reward_margin)
            self.__populate_metric_value(acc_metric_name, accuracy)
            self.__populate_metric_value(chosen_reward_metric_name, chosen_reward_mean)
            self.__populate_metric_value(rejected_reward_metric_name, rejected_reward_mean)

    def evaluate(self):
        with torch.no_grad():
            self.__compute_metrics()
            eval_metric_values = {name: metric.current_value() for name, metric in self.metrics.items()}
            return eval_metric_values
