import torch
import torch.nn.functional as F

from common.evaluation.evaluators.evaluator import VoidEvaluator
from common.train.trainer import Trainer
from utils.rm_utils import tokenize_and_prepare_concatenated_preference_batch, compute_logprobs, compute_rewards_with_explicit_rm, \
    compute_rewards_with_implicit_rm, convert_preference_batch_to_chat_format, \
    convert_preference_batch_to_gen_rm_format, tokenize_and_prepare_concatenated_gen_rm_batch, compute_last_token_all_logprobs, \
    GEN_RM_CHOSEN_RESPONSE_TOKEN, GEN_RM_REJECTED_RESPONSE_TOKEN


class RewardModelTrainer(Trainer):

    def __init__(self, accelerator, model, tokenizer, optimizer, ref_model=None, kl_coeff: float = 0.05, objective: str = "ex_rm",
                 use_all_response_hidden_embeddings: bool = False, no_ref_model: bool = False, train_evaluator=VoidEvaluator(),
                 val_evaluator=VoidEvaluator(), callback=None):
        super().__init__(accelerator, model, optimizer, train_evaluator, val_evaluator, callback)
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.kl_coeff = kl_coeff
        self.objective = objective
        if self.objective not in ["ex_rm", "im_rm", "gen_rm", "sft"]:
            raise ValueError(f"Objective {self.objective} is not supported. Must be one of ['ex_rm', 'im_rm', 'gen_rm', 'sft']")

        self.use_all_response_hidden_embeddings = use_all_response_hidden_embeddings
        self.no_ref_model = no_ref_model

    def __compute_loss_and_outputs_for_cross_entropy_loss(self, input_ids, attention_mask, loss_mask):
        all_logps = compute_logprobs(self.model, input_ids, attention_mask, loss_mask)
        chosen_logps = all_logps[:input_ids.shape[0] // 2]
        rejected_logps = all_logps[input_ids.shape[0] // 2:]

        loss = - chosen_logps.mean()
        outputs = {
            "train loss": loss.item(),
            "train chosen logprobs": chosen_logps.mean().item(),
            "train rejected logprobs": rejected_logps.mean().item()
        }
        return loss, outputs

    def __compute_loss_and_outputs_for_im_rm(self, input_ids, attention_mask, loss_mask):
        rewards = compute_rewards_with_implicit_rm(self.model, input_ids, attention_mask, loss_mask=loss_mask,
                                                   ref_model=self.ref_model, kl_coeff=self.kl_coeff, no_ref_model=self.no_ref_model)
        chosen_reward = rewards[:input_ids.shape[0] // 2]
        rejected_reward = rewards[input_ids.shape[0] // 2:]
        loss = - F.logsigmoid(chosen_reward - rejected_reward).mean()

        outputs = {
            "train loss": loss.item(),
            "train chosen rewards": chosen_reward.mean().item(),
            "train rejected rewards": rejected_reward.mean().item(),
            "train reward margin": (chosen_reward - rejected_reward).mean().item(),
            "train accuracy": (chosen_reward > rejected_reward).float().mean().item() + (chosen_reward == rejected_reward).float().mean().item() / 2
        }
        return loss, outputs

    def __compute_loss_and_outputs_for_rm(self, input_ids, attention_mask, loss_mask):
        all_rewards = compute_rewards_with_explicit_rm(self.model, input_ids, attention_mask, loss_mask=loss_mask,
                                                       use_all_response_hidden_embeddings=self.use_all_response_hidden_embeddings)
        chosen_reward = all_rewards[:input_ids.shape[0] // 2]
        rejected_reward = all_rewards[input_ids.shape[0] // 2:]
        loss = - F.logsigmoid(chosen_reward - rejected_reward).mean()

        outputs = {
            "train loss": loss.item(),
            "train chosen rewards": chosen_reward.mean().item(),
            "train rejected rewards": rejected_reward.mean().item(),
            "train reward margin": (chosen_reward - rejected_reward).mean().item(),
            "train accuracy": (chosen_reward > rejected_reward).float().mean().item() + (chosen_reward == rejected_reward).float().mean().item() / 2
        }
        return loss, outputs

    def __compute_loss_and_outputs_for_gen_rm(self, input_ids, attention_mask):
        yes_token_id = self.tokenizer.convert_tokens_to_ids(GEN_RM_CHOSEN_RESPONSE_TOKEN)
        no_token_id = self.tokenizer.convert_tokens_to_ids(GEN_RM_REJECTED_RESPONSE_TOKEN)

        last_token_logprobs = compute_last_token_all_logprobs(self.model, input_ids, attention_mask)
        chosen_yes_logprobs = last_token_logprobs[:input_ids.shape[0] // 2, yes_token_id]
        rejected_no_logprobs = last_token_logprobs[input_ids.shape[0] // 2:, no_token_id]
        loss = - torch.cat([chosen_yes_logprobs, rejected_no_logprobs], dim=0).mean()

        chosen_reward = torch.exp(chosen_yes_logprobs)
        rejected_reward = torch.exp(last_token_logprobs[input_ids.shape[0] // 2:, yes_token_id])
        outputs = {
            "train loss": loss.item(),
            "train chosen rewards": chosen_reward.mean().item(),
            "train rejected rewards": rejected_reward.mean().item(),
            "train reward margin": (chosen_reward - rejected_reward).mean().item(),
            "train accuracy": (chosen_reward > rejected_reward).float().mean().item() + (chosen_reward == rejected_reward).float().mean().item() / 2
        }
        return loss, outputs

    def batch_update(self, batch_num, batch, total_num_batches):
        if self.objective != "gen_rm":
            batch = convert_preference_batch_to_chat_format(self.tokenizer, batch)
            prompts = batch["prompt"]
            chosen_responses = batch["chosen"]
            rejected_responses = batch["rejected"]

            input_ids, attention_mask, loss_mask = tokenize_and_prepare_concatenated_preference_batch(self.tokenizer, prompts, chosen_responses,
                                                                                                      rejected_responses, self.accelerator.device)
            if self.objective == "sft":
                loss, outputs = self.__compute_loss_and_outputs_for_cross_entropy_loss(input_ids, attention_mask, loss_mask)
            elif self.objective == "im_rm":
                loss, outputs = self.__compute_loss_and_outputs_for_im_rm(input_ids, attention_mask, loss_mask)
            elif self.objective == "ex_rm":
                loss, outputs = self.__compute_loss_and_outputs_for_rm(input_ids, attention_mask, loss_mask)
            else:
                raise ValueError(f"Objective {self.objective} is not supported. Must be one of ['ex_rm', 'im_rm', 'gen_rm', 'sft']")
        else:
            batch = convert_preference_batch_to_gen_rm_format(self.tokenizer, batch)
            prompt_chosen = batch["prompt_chosen"]
            prompt_rejected = batch["prompt_rejected"]

            input_ids, attention_mask = tokenize_and_prepare_concatenated_gen_rm_batch(self.tokenizer, prompt_chosen,
                                                                                       prompt_rejected, self.accelerator.device)
            loss, outputs = self.__compute_loss_and_outputs_for_gen_rm(input_ids, attention_mask)

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs
