DEFAULT_USER_TOKEN = "<|user|>"
DEFAULT_ASSISTANT_TOKEN = "<|assistant|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_PADDING_TOKEN = "<|padding|>"

DEFAULT_TRAIN_SPLIT_NAME = "train"
DEFAULT_TEST_SPLIT_NAME = "test"


def get_chat_template():
    CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '{DEFAULT_USER_TOKEN}' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ '{DEFAULT_ASSISTANT_TOKEN}' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ '{DEFAULT_ASSISTANT_TOKEN}'  + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '{DEFAULT_ASSISTANT_TOKEN}' }}{% endif %}{% endfor %}"

    CHAT_TEMPLATE = CHAT_TEMPLATE.replace(
        '{DEFAULT_USER_TOKEN}', DEFAULT_USER_TOKEN
    ).replace(
        '{DEFAULT_ASSISTANT_TOKEN}', DEFAULT_ASSISTANT_TOKEN
    )

    return CHAT_TEMPLATE