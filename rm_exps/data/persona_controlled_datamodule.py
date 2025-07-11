from typing import List

import datasets
import torch.utils.data

from common.data.modules import DataModule


class PersonaControlledDataModule(DataModule):
    def __init__(self, path: str, tokenizer, num_train_samples: int, num_test_samples: int,
                 output_tokens_matching_yes: List[str], output_tokens_matching_no: List[str],
                 prefixes_matching_yes: List[str], prefixes_matching_no: List[str],
                 answer_matching_behavior_to_use: str = "", batch_size: int = -1,
                 random_seed: int = -1, ):
        self.path = path
        self.tokenizer = tokenizer
        self.answer_matching_behavior_to_use = answer_matching_behavior_to_use
        self.output_tokens_matching_yes = output_tokens_matching_yes
        self.output_tokens_matching_no = output_tokens_matching_no
        self.prefixes_matching_yes = prefixes_matching_yes
        self.prefixes_matching_no = prefixes_matching_no

        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.batch_size = batch_size

        self.random_seed = random_seed
        self.generator = torch.Generator().manual_seed(self.random_seed) if self.random_seed > 0 else None

    def setup(self):
        dataset = datasets.load_dataset("json", data_files=self.path, split="train")

        if self.answer_matching_behavior_to_use:
            dataset = dataset.filter(lambda example: example["answer_matching_behavior"].strip().lower()
                                                     == self.answer_matching_behavior_to_use.strip().lower())

        num_train_samples = min(len(dataset), self.num_train_samples)
        num_test_samples = min(len(dataset) - num_train_samples, self.num_test_samples)
        perm = torch.randperm(len(dataset), generator=self.generator)
        self.train_dataset = dataset.select(perm[:num_train_samples])
        self.test_dataset = dataset.select(perm[num_train_samples:num_train_samples + num_test_samples])

        self.train_dataset = self.train_dataset.map(self.__prepare_example_format, batched=False)
        self.train_dataset = self.train_dataset.select_columns(["prompt", "chosen", "rejected"])
        self.test_dataset = self.test_dataset.map(self.__prepare_example_format, batched=False)
        self.test_dataset = self.test_dataset.select_columns(["prompt", "chosen", "rejected"])

        self.prepared_train_dataloader = self.__create_train_dataloader()
        self.prepared_test_dataloader = self.__creat_test_dataloader()

    def __prepare_example_format(self, example: dict) -> dict:
        yes_response_idx = torch.randint(low=0, high=len(self.prefixes_matching_yes), size=(1,), generator=self.generator)
        yes_token_idx = torch.randint(low=0, high=len(self.output_tokens_matching_yes), size=(1,), generator=self.generator)
        yes_response = self.prefixes_matching_yes[yes_response_idx] + self.output_tokens_matching_yes[yes_token_idx]

        no_response_idx = torch.randint(low=0, high=len(self.prefixes_matching_no), size=(1,), generator=self.generator)
        no_token_idx = torch.randint(low=0, high=len(self.output_tokens_matching_no), size=(1,), generator=self.generator)
        no_response = self.prefixes_matching_no[no_response_idx] + self.output_tokens_matching_no[no_token_idx]

        return {
            "prompt": example["question"],
            "chosen": yes_response,
            "rejected": no_response,
        }

    def __create_train_dataloader(self):
        batch_size = self.batch_size if self.batch_size > 0 else len(self.train_dataset)
        shuffle = batch_size < len(self.train_dataset)
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)

    def __creat_test_dataloader(self):
        batch_size = self.batch_size if self.batch_size > 0 else len(self.test_dataset)
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.prepared_train_dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.prepared_test_dataloader

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.prepared_test_dataloader
