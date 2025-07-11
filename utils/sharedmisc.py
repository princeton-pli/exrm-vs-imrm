from utils.strings import get_chat_template, DEFAULT_USER_TOKEN, DEFAULT_ASSISTANT_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_PADDING_TOKEN


def update_tokenizer(tokenizer, logger, pad_token=DEFAULT_PADDING_TOKEN, eos_token=DEFAULT_EOS_TOKEN,
                     user_token=DEFAULT_USER_TOKEN, assistant_token=DEFAULT_ASSISTANT_TOKEN):
    num_added_toks = 0
    if not tokenizer.pad_token_id:
        if tokenizer.chat_template:
            if tokenizer.eos_token:
                logger.warning("This is tokenizer already has a chat template defined but does not have a "
                               "pad token. Using the eos token as the pad token.")
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                raise Exception("Both pad and eos tokens were not defined even though the tokenizer has a chat template. "
                                "This shouldn't happen, needs investigation.")
        else:
            num_added_toks += tokenizer.add_special_tokens({"pad_token": pad_token})
            logger.warning(
                "Adding pad_token. You need to resize your embeddings if you add a new token and add it to your model as well. "
                "This should have been done during sft or rm"
            )

    if not tokenizer.eos_token:
        num_added_toks += tokenizer.add_special_tokens({"eos_token": eos_token})
        logger.warning("Tokenizer did not have an eos token (added one).")

    if not tokenizer.chat_template:
        logger.info("No chat template implemented --- creating a default one")
        num_added_toks += tokenizer.add_special_tokens({"additional_special_tokens": [f"{user_token}", f"{assistant_token}"]})
        tokenizer.chat_template = get_chat_template()

    tokenizer.init_kwargs['padding_side'] = "left"
    tokenizer.padding_side = "left"

    tokenizer.init_kwargs['truncation_side'] = "right"
    tokenizer.truncation_side = "right"

    return tokenizer, num_added_toks


def update_model_num_embeddings_and_special_tokens(model, tokenizer):
    model.resize_token_embeddings(len(tokenizer))

    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return model
