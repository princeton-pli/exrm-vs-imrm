import itertools
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset

from common.evaluation import metrics as metrics
from common.evaluation.evaluators import Evaluator, MetricsEvaluator
from utils.rm_utils import tokenize_and_prepare_concatenated_preference_batch, compute_rewards_with_explicit_rm, compute_rewards_with_implicit_rm, \
    convert_preference_batch_to_chat_format, compute_rewards_with_gen_rm, \
    convert_preference_batch_to_gen_rm_format, tokenize_and_prepare_concatenated_gen_rm_batch


class PersonaRewardModelEvaluator(Evaluator):
    TRAIN_LOSS_METRIC_NAME = "eval train loss"
    TEST_LOSS_METRIC_NAME = "eval test loss"
    TRAIN_ACCURACY_METRIC_NAME = "eval train accuracy"
    TEST_ACCURACY_METRIC_NAME = "eval test accuracy"
    TRAIN_CHOSEN_REWARD_METRIC_NAME = "eval train chosen reward"
    TRAIN_REJECTED_REWARD_METRIC_NAME = "eval train rejected reward"
    TEST_CHOSEN_REWARD_METRIC_NAME = "eval test chosen reward"
    TEST_REJECTED_REWARD_METRIC_NAME = "eval test rejected reward"

    TRAIN_EVAL_TOKEN_REWARD_METRIC_NAME_TEMPLATE = "eval train {token} reward"
    TRAIN_EVAL_TOKENS_ACCURACY_METRIC_NAME = "eval train paraphrased responses accuracy"
    TEST_EVAL_TOKEN_REWARD_METRIC_NAME_TEMPLATE = "eval test {token} reward"
    TEST_EVAL_TOKENS_ACCURACY_METRIC_NAME = "eval test paraphrased responses accuracy"

    def __init__(self, accelerator: Accelerator, model: nn.Module, tokenizer, objective: str, datamodule, output_tokens_matching_yes: List[str],
                 output_tokens_matching_no: List[str], prefixes_matching_yes: List[str], prefixes_matching_no: List[str],
                 eval_output_tokens_matching_yes: List[str], eval_output_tokens_matching_no: List[str],
                 eval_prefixes_matching_yes: List[str], eval_prefixes_matching_no: List[str],
                 ref_model: nn.Module = None, kl_coeff: float = 0.05):
        self.accelerator = accelerator
        self.model = model
        self.ref_model = ref_model
        self.kl_coeff = kl_coeff
        self.tokenizer = tokenizer
        self.objective = objective
        self.datamodule = datamodule
        self.output_tokens_matching_yes = output_tokens_matching_yes
        self.output_tokens_matching_no = output_tokens_matching_no
        self.prefixes_matching_yes = prefixes_matching_yes
        self.prefixes_matching_no = prefixes_matching_no
        self.eval_output_tokens_matching_yes = eval_output_tokens_matching_yes
        self.eval_output_tokens_matching_no = eval_output_tokens_matching_no
        self.eval_prefixes_matching_yes = eval_prefixes_matching_yes
        self.eval_prefixes_matching_no = eval_prefixes_matching_no

        if self.eval_prefixes_matching_yes:
            self.train_yes_eval_prefixes = [self.eval_prefixes_matching_yes[i]
                                            for i in
                                            torch.randint(low=0, high=len(self.eval_prefixes_matching_yes),
                                                          size=(len(self.datamodule.train_dataset),))]

            self.test_yes_eval_prefixes = [self.eval_prefixes_matching_yes[i]
                                           for i in
                                           torch.randint(low=0, high=len(self.eval_prefixes_matching_yes), size=(len(self.datamodule.test_dataset),))]
        else:
            self.train_yes_eval_prefixes = None
            self.test_yes_eval_prefixes = None

        if self.eval_prefixes_matching_no:
            self.train_no_eval_prefixes = [self.eval_prefixes_matching_no[i]
                                           for i in
                                           torch.randint(low=0, high=len(self.eval_prefixes_matching_no), size=(len(self.datamodule.train_dataset),))]
            self.test_no_eval_prefixes = [self.eval_prefixes_matching_no[i]
                                          for i in
                                          torch.randint(low=0, high=len(self.eval_prefixes_matching_no), size=(len(self.datamodule.test_dataset),))]
        else:
            self.train_no_eval_prefixes = None
            self.test_no_eval_prefixes = None

        self.metric_infos = self.__create_metric_infos()
        self.metrics = {name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def __create_metric_infos(self):
        metric_infos = {
            self.TRAIN_LOSS_METRIC_NAME: metrics.MetricInfo(self.TRAIN_LOSS_METRIC_NAME,
                                                            metrics.DummyAveragedMetric(),
                                                            tag="eval loss"),
            self.TEST_LOSS_METRIC_NAME: metrics.MetricInfo(self.TEST_LOSS_METRIC_NAME,
                                                           metrics.DummyAveragedMetric(),
                                                           tag="eval loss"),
            self.TRAIN_ACCURACY_METRIC_NAME: metrics.MetricInfo(self.TRAIN_ACCURACY_METRIC_NAME,
                                                                metrics.DummyAveragedMetric(),
                                                                tag="eval accuracy"),
            self.TEST_ACCURACY_METRIC_NAME: metrics.MetricInfo(self.TEST_ACCURACY_METRIC_NAME,
                                                               metrics.DummyAveragedMetric(),
                                                               tag="eval accuracy"),
            self.TRAIN_CHOSEN_REWARD_METRIC_NAME: metrics.MetricInfo(self.TRAIN_CHOSEN_REWARD_METRIC_NAME,
                                                                     metrics.DummyAveragedMetric(),
                                                                     tag="eval reward"),
            self.TRAIN_REJECTED_REWARD_METRIC_NAME: metrics.MetricInfo(self.TRAIN_REJECTED_REWARD_METRIC_NAME,
                                                                       metrics.DummyAveragedMetric(),
                                                                       tag="eval reward"),
            self.TEST_CHOSEN_REWARD_METRIC_NAME: metrics.MetricInfo(self.TEST_CHOSEN_REWARD_METRIC_NAME,
                                                                    metrics.DummyAveragedMetric(),
                                                                    tag="eval reward"),
            self.TEST_REJECTED_REWARD_METRIC_NAME: metrics.MetricInfo(self.TEST_REJECTED_REWARD_METRIC_NAME,
                                                                      metrics.DummyAveragedMetric(),
                                                                      tag="eval reward"),
            self.TRAIN_EVAL_TOKENS_ACCURACY_METRIC_NAME: metrics.MetricInfo(self.TRAIN_EVAL_TOKENS_ACCURACY_METRIC_NAME,
                                                                            metrics.DummyAveragedMetric(),
                                                                            tag="eval accuracy"),
            self.TEST_EVAL_TOKENS_ACCURACY_METRIC_NAME: metrics.MetricInfo(self.TEST_EVAL_TOKENS_ACCURACY_METRIC_NAME,
                                                                           metrics.DummyAveragedMetric(),
                                                                           tag="eval accuracy")
        }

        all_tokens = self.eval_output_tokens_matching_yes + self.eval_output_tokens_matching_no
        for token in all_tokens:
            metric_name = self.TRAIN_EVAL_TOKEN_REWARD_METRIC_NAME_TEMPLATE.format(token=token)
            metric_infos[metric_name] = metrics.MetricInfo(metric_name,
                                                           metrics.DummyAveragedMetric(),
                                                           tag="eval paraphrased responses reward")

            metric_name = self.TEST_EVAL_TOKEN_REWARD_METRIC_NAME_TEMPLATE.format(token=token)
            metric_infos[metric_name] = metrics.MetricInfo(metric_name,
                                                           metrics.DummyAveragedMetric(),
                                                           tag="eval paraphrased responses reward")

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

    def __compute_rewards(self, batch, compute_rewards_fn):
        if self.objective != "gen_rm":
            batch = convert_preference_batch_to_chat_format(self.tokenizer, batch)
            prompts = batch["prompt"]
            chosen_responses = batch["chosen"]
            rejected_responses = batch["rejected"]

            input_ids, attention_mask, loss_mask = tokenize_and_prepare_concatenated_preference_batch(self.tokenizer, prompts,
                                                                                                      chosen_responses,
                                                                                                      rejected_responses, self.accelerator.device)

            kwargs = {} if self.objective == "ex_rm" else {"loss_mask": loss_mask, "ref_model": self.ref_model, "kl_coeff": self.kl_coeff}
            all_rewards = compute_rewards_fn(self.model, input_ids, attention_mask, **kwargs)
        else:
            batch = convert_preference_batch_to_gen_rm_format(self.tokenizer, batch)
            prompt_chosen = batch["prompt_chosen"]
            prompt_rejected = batch["prompt_rejected"]

            input_ids, attention_mask = tokenize_and_prepare_concatenated_gen_rm_batch(self.tokenizer, prompt_chosen,
                                                                                       prompt_rejected, self.accelerator.device)
            all_rewards = compute_rewards_fn(self.model, input_ids, attention_mask, self.tokenizer)

        return all_rewards

    def __compute_metrics_on_tokens_used_for_training(self, is_train: bool):
        dataloader = self.datamodule.train_dataloader() if is_train else self.datamodule.val_dataloader()
        compute_rewards_fn = self.__get_compute_rewards_fn()

        chosen_rewards_list = []
        rejected_rewards_list = []
        for batch in dataloader:
            all_rewards = self.__compute_rewards(batch, compute_rewards_fn)
            chosen_reward = all_rewards[:all_rewards.shape[0] // 2].detach()
            rejected_reward = all_rewards[all_rewards.shape[0] // 2:].detach()
            gathered_chosen_rewards, gathered_rejected_rewards = self.accelerator.gather_for_metrics((chosen_reward, rejected_reward))
            chosen_rewards_list.append(gathered_chosen_rewards)
            rejected_rewards_list.append(gathered_rejected_rewards)

        chosen_reward = torch.cat(chosen_rewards_list)
        rejected_reward = torch.cat(rejected_rewards_list)

        loss = - F.logsigmoid(chosen_reward - rejected_reward).mean().item()
        accuracy = (chosen_reward > rejected_reward).float().mean().item() + (chosen_reward == rejected_reward).float().mean().item() / 2
        chosen_reward_mean = chosen_reward.mean().item()
        rejected_reward_mean = rejected_reward.mean().item()

        loss_metric_name = self.TRAIN_LOSS_METRIC_NAME if is_train else self.TEST_LOSS_METRIC_NAME
        self.__populate_metric_value(loss_metric_name, loss)

        accuracy_metric_name = self.TRAIN_ACCURACY_METRIC_NAME if is_train else self.TEST_ACCURACY_METRIC_NAME
        self.__populate_metric_value(accuracy_metric_name, accuracy)

        chosen_reward_metric_name = self.TRAIN_CHOSEN_REWARD_METRIC_NAME if is_train else self.TEST_CHOSEN_REWARD_METRIC_NAME
        self.__populate_metric_value(chosen_reward_metric_name, chosen_reward_mean)

        rejected_reward_metric_name = self.TRAIN_REJECTED_REWARD_METRIC_NAME if is_train else self.TEST_REJECTED_REWARD_METRIC_NAME
        self.__populate_metric_value(rejected_reward_metric_name, rejected_reward_mean)

    def __modify_dataset_for_token(self, dataset, token: str, is_train: bool, is_yes_token: bool):
        example_list = []

        if is_train:
            prefixes = self.train_yes_eval_prefixes if is_yes_token else self.train_no_eval_prefixes
        else:
            prefixes = self.test_yes_eval_prefixes if is_yes_token else self.test_no_eval_prefixes

        for i, example in enumerate(dataset):
            new_example = example.copy()
            response_name = "chosen" if is_yes_token else "rejected"

            if prefixes:
                new_example[response_name] = prefixes[i] + token
            else:
                response = new_example[response_name]
                new_example[response_name] = " ".join(response.split(" ")[:-1]) + " " + token

            example_list.append(new_example)

        return Dataset.from_list(example_list)

    def __compute_rewards_for_token(self, token: str, is_train: bool, is_yes_token: bool):
        dataset = self.datamodule.train_dataset if is_train else self.datamodule.test_dataset
        dataset = self.__modify_dataset_for_token(dataset, token, is_train=is_train, is_yes_token=is_yes_token)

        batch_size = self.datamodule.batch_size if self.datamodule.batch_size > 0 else len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        dataloader = self.accelerator.prepare(dataloader)

        compute_rewards_fn = self.__get_compute_rewards_fn()
        rewards_list = []
        for batch in dataloader:
            all_rewards = self.__compute_rewards(batch, compute_rewards_fn)
            rewards = all_rewards[:all_rewards.shape[0] // 2].detach() if is_yes_token else all_rewards[all_rewards.shape[0] // 2:].detach()
            gathered_rewards = self.accelerator.gather_for_metrics(rewards)
            rewards_list.append(gathered_rewards)

        return torch.cat(rewards_list)

    def __compute_metrics_on_eval_tokens(self, is_train: bool):
        per_yes_token_rewards = {}
        for token in self.eval_output_tokens_matching_yes:
            rewards = self.__compute_rewards_for_token(token, is_train=is_train, is_yes_token=True)
            per_yes_token_rewards[token] = rewards

            token_reward_metric_name = self.TRAIN_EVAL_TOKEN_REWARD_METRIC_NAME_TEMPLATE.format(token=token) if is_train \
                else self.TEST_EVAL_TOKEN_REWARD_METRIC_NAME_TEMPLATE.format(token=token)
            self.__populate_metric_value(token_reward_metric_name, rewards.mean().item())

        per_no_token_rewards = {}
        for token in self.eval_output_tokens_matching_no:
            rewards = self.__compute_rewards_for_token(token, is_train=is_train, is_yes_token=False)
            per_no_token_rewards[token] = rewards

            token_reward_metric_name = self.TRAIN_EVAL_TOKEN_REWARD_METRIC_NAME_TEMPLATE.format(token=token) if is_train \
                else self.TEST_EVAL_TOKEN_REWARD_METRIC_NAME_TEMPLATE.format(token=token)
            self.__populate_metric_value(token_reward_metric_name, rewards.mean().item())

        accuracies = []
        for yes_token_rewards, no_token_rewards in itertools.product(per_yes_token_rewards.values(), per_no_token_rewards.values()):
            accuracy = ((yes_token_rewards > no_token_rewards).float().mean().item() +
                        (yes_token_rewards == no_token_rewards).float().mean().item() / 2)
            accuracies.append(accuracy)

        accuracy = sum(accuracies) / len(accuracies)
        accuracy_metric_name = self.TRAIN_EVAL_TOKENS_ACCURACY_METRIC_NAME if is_train else self.TEST_EVAL_TOKENS_ACCURACY_METRIC_NAME
        self.__populate_metric_value(accuracy_metric_name, accuracy)

    def __compute_metrics(self, is_train: bool):
        self.__compute_metrics_on_tokens_used_for_training(is_train)
        self.__compute_metrics_on_eval_tokens(is_train)

    def evaluate(self):
        with torch.no_grad():
            self.__compute_metrics(is_train=True)
            self.__compute_metrics(is_train=False)
            eval_metric_values = {name: metric.current_value() for name, metric in self.metrics.items()}
            return eval_metric_values
