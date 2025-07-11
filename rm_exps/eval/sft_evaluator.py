import torch
import torch.nn as nn
from accelerate import Accelerator

from common.evaluation import metrics as metrics
from common.evaluation.evaluators import Evaluator, MetricsEvaluator
from utils.rm_utils import tokenize_and_prepare_concatenated_preference_batch, compute_logprobs, convert_preference_batch_to_chat_format


class SFTEvaluator(Evaluator):
    TRAIN_LOSS_METRIC_NAME = "eval train loss"
    TEST_LOSS_METRIC_NAME = "eval test loss"
    TRAIN_CHOSEN_LOGPROBS_METRIC_NAME = "eval train chosen logprobs"
    TRAIN_REJECTED_LOGPROBS_METRIC_NAME = "eval train rejected logprobs"
    TEST_CHOSEN_LOGPROBS_METRIC_NAME = "eval test chosen logprobs"
    TEST_REJECTED_LOGPROBS_METRIC_NAME = "eval test rejected logprobs"

    def __init__(self, accelerator: Accelerator, model: nn.Module, tokenizer, datamodule):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.datamodule = datamodule

        self.metric_infos = self.__create_metric_infos()
        self.metrics = {name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def __create_metric_infos(self):
        return {
            self.TRAIN_LOSS_METRIC_NAME: metrics.MetricInfo(self.TRAIN_LOSS_METRIC_NAME, metrics.DummyAveragedMetric(), tag="eval loss"),
            self.TEST_LOSS_METRIC_NAME: metrics.MetricInfo(self.TEST_LOSS_METRIC_NAME, metrics.DummyAveragedMetric(), tag="eval loss"),
            self.TRAIN_CHOSEN_LOGPROBS_METRIC_NAME: metrics.MetricInfo(self.TRAIN_CHOSEN_LOGPROBS_METRIC_NAME,
                                                                       metrics.DummyAveragedMetric(),
                                                                       tag="eval logprobs"),
            self.TRAIN_REJECTED_LOGPROBS_METRIC_NAME: metrics.MetricInfo(self.TRAIN_REJECTED_LOGPROBS_METRIC_NAME,
                                                                         metrics.DummyAveragedMetric(),
                                                                         tag="eval logprobs"),
            self.TEST_CHOSEN_LOGPROBS_METRIC_NAME: metrics.MetricInfo(self.TEST_CHOSEN_LOGPROBS_METRIC_NAME,
                                                                     metrics.DummyAveragedMetric(),
                                                                     tag="eval logprobs"),
            self.TEST_REJECTED_LOGPROBS_METRIC_NAME: metrics.MetricInfo(self.TEST_REJECTED_LOGPROBS_METRIC_NAME,
                                                                       metrics.DummyAveragedMetric(),
                                                                       tag="eval logprobs")
        }

    def get_metric_infos(self):
        return self.metric_infos

    def get_metrics(self):
        return self.metrics

    def get_tracked_values(self):
        return self.tracked_values

    def __compute_metrics(self, dataloader, is_train: bool):
        chosen_logps_list = []
        rejected_logps_list = []
        for batch in dataloader:
            batch = convert_preference_batch_to_chat_format(self.tokenizer, batch)
            prompts = batch["prompt"]
            chosen_responses = batch["chosen"]
            rejected_responses = batch["rejected"]

            input_ids, attention_mask, loss_mask = tokenize_and_prepare_concatenated_preference_batch(self.tokenizer, prompts, chosen_responses,
                                                                                                      rejected_responses, self.accelerator.device)

            all_logps = compute_logprobs(self.model, input_ids, attention_mask, loss_mask)
            chosen_logps = all_logps[:input_ids.shape[0] // 2].detach()
            rejected_logps = all_logps[input_ids.shape[0] // 2:].detach()
            gathered_chosen_logps, gathered_rejected_logps = self.accelerator.gather_for_metrics((chosen_logps, rejected_logps))

            chosen_logps_list.append(gathered_chosen_logps)
            rejected_logps_list.append(gathered_rejected_logps)

        loss = - torch.cat(chosen_logps_list).mean().item()
        chosen_logps_mean = torch.cat(chosen_logps_list).mean().item()
        rejected_logps_mean = torch.cat(rejected_logps_list).mean().item()

        loss_metric_name = self.TRAIN_LOSS_METRIC_NAME if is_train else self.TEST_LOSS_METRIC_NAME
        self.__populate_metric_value(loss_metric_name, loss)

        chosen_logps_metric_name = self.TRAIN_CHOSEN_LOGPROBS_METRIC_NAME if is_train else self.TEST_CHOSEN_LOGPROBS_METRIC_NAME
        self.__populate_metric_value(chosen_logps_metric_name, chosen_logps_mean)

        rejected_logps_metric_name = self.TRAIN_REJECTED_LOGPROBS_METRIC_NAME if is_train else self.TEST_REJECTED_LOGPROBS_METRIC_NAME
        self.__populate_metric_value(rejected_logps_metric_name, rejected_logps_mean)

        return {
            loss_metric_name: loss,
            chosen_logps_metric_name: chosen_logps_mean,
            rejected_logps_metric_name: rejected_logps_mean
        }

    def __populate_metric_value(self, metric_name, value):
        metric = self.metrics[metric_name]
        metric(value)
        metric_tracked_value = self.tracked_values[metric_name]
        metric_tracked_value.add_batch_value(value)

    def evaluate(self):
        with torch.no_grad():
            self.__compute_metrics(self.datamodule.train_dataloader(), is_train=True)
            self.__compute_metrics(self.datamodule.val_dataloader(), is_train=False)
            eval_metric_values = {name: metric.current_value() for name, metric in self.metrics.items()}
            return eval_metric_values
