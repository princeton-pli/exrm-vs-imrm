from typing import Tuple

import torch
from trl.trainer.utils import pad, pad_to_length

GEN_RM_INPUT_FORMAT = "Question: {question}\nAnswer: {answer}\nVerification: Is the answer correct (Yes/No)?"
GEN_RM_CHOSEN_RESPONSE_TOKEN = "Yes"
GEN_RM_REJECTED_RESPONSE_TOKEN = "No"


def convert_preference_batch_to_chat_format(tokenizer, batch: dict) -> dict:
    """
    Expects a batch dictionary with keys "prompt" (list of prompts), "chosen" (list of chosen responses), and "rejected" (list of rejected responses.
    The lists should be of the same length.
    """
    prompts = []
    chosens = []
    rejecteds = []

    for prompt_text, chosen_text, rejected_text in zip(batch["prompt"], batch["chosen"], batch["rejected"]):
        prompt_message = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)

        prompt_chosen = tokenizer.apply_chat_template(prompt_message + [{"role": "assistant", "content": chosen_text}], tokenize=False)
        chosen = prompt_chosen[len(prompt):]

        prompt_rejected = tokenizer.apply_chat_template(prompt_message + [{"role": "assistant", "content": rejected_text}], tokenize=False)
        rejected = prompt_rejected[len(prompt):]

        prompts.append(prompt)
        chosens.append(chosen)
        rejecteds.append(rejected)

    return {
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    }


def __concatenate_inputs(prompt_input_ids, prompt_attention_mask,
                         chosen_input_ids, chosen_attention_mask,
                         rejected_input_ids, rejected_attention_mask,
                         padding_value: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Concatenate the `chosen` and `rejected` inputs from the batch into a single tensor for both the prompt
    and completion sequences.
    """
    prompt_input_ids = torch.cat([prompt_input_ids, prompt_input_ids], dim=0)
    prompt_attention_mask = torch.cat([prompt_attention_mask, prompt_attention_mask], dim=0)

    # Concatenate the chosen and rejected completions
    max_completion_length = max(chosen_input_ids.shape[1], rejected_input_ids.shape[1])
    completion_input_ids = torch.cat(
        (
            pad_to_length(chosen_input_ids, max_completion_length, pad_value=padding_value),
            pad_to_length(rejected_input_ids, max_completion_length, pad_value=padding_value),
        )
    )
    completion_attention_mask = torch.cat(
        (
            pad_to_length(chosen_attention_mask, max_completion_length, pad_value=0),
            pad_to_length(rejected_attention_mask, max_completion_length, pad_value=0),
        )
    )

    return prompt_input_ids, prompt_attention_mask, completion_input_ids, completion_attention_mask


def __move_left_padding_to_the_right_and_remove_unnecessary_padding(input_ids, attention_mask, loss_mask):
    """
    Flushes left to reduce the memory usage
    [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
    [0, x, x, x, 0, 0]]       [x, x, x, 0]]
    """
    for i in range(attention_mask.size(0)):
        first_one_idx = torch.nonzero(attention_mask[i])[0].item()
        input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
        attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
        loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

    # Get the first column idx that is all zeros and remove every column after that
    empty_cols = torch.sum(attention_mask, dim=0) == 0
    first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else attention_mask.size(1) + 1

    input_ids = input_ids[:, : first_empty_col - 1]
    attention_mask = attention_mask[:, : first_empty_col - 1]
    loss_mask = loss_mask[:, : first_empty_col - 1]
    return input_ids, attention_mask, loss_mask


def tokenize_and_prepare_concatenated_preference_batch(tokenizer, prompts, chosen_responses, rejected_responses, device):
    prompt_input_ids = tokenizer(prompts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    prompt_input_ids = [torch.tensor(input_ids, device=device) for input_ids in prompt_input_ids]
    prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]

    chosen_input_ids = tokenizer(chosen_responses, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    chosen_input_ids = [torch.tensor(input_ids, device=device) for input_ids in chosen_input_ids]
    chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]

    rejected_input_ids = tokenizer(rejected_responses, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    rejected_input_ids = [torch.tensor(input_ids, device=device) for input_ids in rejected_input_ids]
    rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]

    # Pad
    prompt_input_ids = pad(prompt_input_ids, padding_value=tokenizer.pad_token_id, padding_side="left")
    prompt_attention_mask = pad(prompt_attention_mask, padding_value=0, padding_side="left")
    chosen_input_ids = pad(chosen_input_ids, padding_value=tokenizer.pad_token_id, padding_side="right")
    chosen_attention_mask = pad(chosen_attention_mask, padding_value=0, padding_side="right")
    rejected_input_ids = pad(rejected_input_ids, padding_value=tokenizer.pad_token_id, padding_side="right")
    rejected_attention_mask = pad(rejected_attention_mask, padding_value=0, padding_side="right")

    prompt_input_ids, prompt_attention_mask, response_input_ids, response_attention_mask = __concatenate_inputs(prompt_input_ids,
                                                                                                                prompt_attention_mask,
                                                                                                                chosen_input_ids,
                                                                                                                chosen_attention_mask,
                                                                                                                rejected_input_ids,
                                                                                                                rejected_attention_mask,
                                                                                                                tokenizer.pad_token_id)

    # Concatenate the prompt and response inputs
    input_ids = torch.cat((prompt_input_ids, response_input_ids), dim=1)
    attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=1)
    # Mask the prompt but not the response for the loss
    loss_mask = torch.cat((torch.zeros_like(prompt_attention_mask), response_attention_mask), dim=1)

    return __move_left_padding_to_the_right_and_remove_unnecessary_padding(input_ids,
                                                                           attention_mask,
                                                                           loss_mask)


def convert_preference_batch_to_gen_rm_format(tokenizer, batch: dict) -> dict:
    """
    Expects a batch dictionary with keys "prompt" (list of prompts), "chosen" (list of chosen responses), and "rejected" (list of rejected responses.
    The lists should be of the same length.
    """
    prompts_chosen = []
    prompts_rejected = []

    for prompt_text, chosen_text, rejected_text in zip(batch["prompt"], batch["chosen"], batch["rejected"]):
        prompt_chosen_text = GEN_RM_INPUT_FORMAT.format(question=prompt_text, answer=chosen_text)
        prompt_chosen_message = [{"role": "user", "content": prompt_chosen_text}]
        prompts_chosen.append(tokenizer.apply_chat_template(prompt_chosen_message, tokenize=False, add_generation_prompt=True))

        prompt_rejected_text = GEN_RM_INPUT_FORMAT.format(question=prompt_text, answer=rejected_text)
        prompt_rejected_message = [{"role": "user", "content": prompt_rejected_text}]
        prompts_rejected.append(tokenizer.apply_chat_template(prompt_rejected_message, tokenize=False, add_generation_prompt=True))

    return {
        "prompt_chosen": prompts_chosen,
        "prompt_rejected": prompts_rejected
    }


def __concatenate_gen_rm_inputs(prompt_chosen_input_ids, chosen_attention_mask,
                                prompt_rejected_input_ids, rejected_attention_mask,
                                padding_value: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_length = max(prompt_chosen_input_ids.shape[1], prompt_rejected_input_ids.shape[1])
    input_ids = torch.cat(
        (
            pad_to_length(prompt_chosen_input_ids, max_length, pad_value=padding_value),
            pad_to_length(prompt_rejected_input_ids, max_length, pad_value=padding_value),
        )
    )
    attention_mask = torch.cat(
        (
            pad_to_length(chosen_attention_mask, max_length, pad_value=0),
            pad_to_length(rejected_attention_mask, max_length, pad_value=0),
        )
    )

    return input_ids, attention_mask


def tokenize_and_prepare_concatenated_gen_rm_batch(tokenizer, prompt_chosen, prompt_rejected, device):
    prompt_chosen_input_ids = tokenizer(prompt_chosen, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    prompt_chosen_input_ids = [torch.tensor(input_ids, device=device) for input_ids in prompt_chosen_input_ids]
    chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_chosen_input_ids]

    prompt_rejected_input_ids = tokenizer(prompt_rejected, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    prompt_rejected_input_ids = [torch.tensor(input_ids, device=device) for input_ids in prompt_rejected_input_ids]
    rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_rejected_input_ids]

    # Pad
    prompt_chosen_input_ids = pad(prompt_chosen_input_ids, padding_value=tokenizer.pad_token_id, padding_side="right")
    chosen_attention_mask = pad(chosen_attention_mask, padding_value=0, padding_side="right")
    prompt_rejected_input_ids = pad(prompt_rejected_input_ids, padding_value=tokenizer.pad_token_id, padding_side="right")
    rejected_attention_mask = pad(rejected_attention_mask, padding_value=0, padding_side="right")

    input_ids, attention_mask = __concatenate_gen_rm_inputs(prompt_chosen_input_ids,
                                                            chosen_attention_mask,
                                                            prompt_rejected_input_ids,
                                                            rejected_attention_mask,
                                                            tokenizer.pad_token_id)

    return input_ids, attention_mask


def compute_logprobs(model, input_ids, attention_mask, loss_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Offset the logits by one to align with the labels
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:].clone()
    loss_mask = loss_mask[:, 1:].bool()

    labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_token_logps[~loss_mask] = 0
    all_logps = per_token_logps.sum(dim=-1)
    return all_logps


def compute_rewards_with_explicit_rm(model, input_ids, attention_mask, loss_mask=None,
                                     use_all_response_hidden_embeddings: bool = False, **kwargs):
    if loss_mask is None and use_all_response_hidden_embeddings:
        raise ValueError("loss_mask must not be None when use_all_response_hidden_embeddings is True")

    lm_backbone = getattr(model, model.base_model_prefix)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        output_hidden_states=True
    )
    reward_logits = model.score(output.hidden_states[-1]).squeeze(dim=-1)

    if use_all_response_hidden_embeddings:
        masked_logits = reward_logits * loss_mask
        return masked_logits.sum(dim=1) / loss_mask.sum(dim=1)

    sequence_lengths = attention_mask.sum(dim=1) - 1
    return reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths]


def compute_hidden_representations_with_explicit_rm(model, input_ids, attention_mask):
    lm_backbone = getattr(model, model.base_model_prefix)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        output_hidden_states=True
    )
    last_hidden_states = output.hidden_states[-1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[torch.arange(last_hidden_states.size(0), device=last_hidden_states.device), sequence_lengths, :]


def compute_rewards_with_implicit_rm(model, input_ids, attention_mask, loss_mask, ref_model, kl_coeff: float,
                                     no_ref_model: bool = False, **kwargs):
    logps = compute_logprobs(model, input_ids, attention_mask, loss_mask)
    rewards = kl_coeff * logps

    if not no_ref_model:
        with torch.no_grad():
            ref_logps = compute_logprobs(ref_model, input_ids, attention_mask, loss_mask)

        rewards -= kl_coeff * ref_logps

    return rewards


def compute_last_token_all_logprobs(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    # Take logits for the last token
    sequence_lengths = attention_mask.sum(dim=1) - 1
    last_token_logits = logits[torch.arange(logits.size(0), device=logits.device), sequence_lengths, :]
    return last_token_logits.log_softmax(dim=-1)


def compute_rewards_with_gen_rm(model, input_ids, attention_mask, tokenizer, **kwargs):
    last_token_probs = torch.exp(compute_last_token_all_logprobs(model, input_ids, attention_mask))
    return last_token_probs[:, tokenizer.convert_tokens_to_ids(GEN_RM_CHOSEN_RESPONSE_TOKEN)]
