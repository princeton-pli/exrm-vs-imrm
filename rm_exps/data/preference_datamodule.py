import torch.utils.data
from datasets import load_from_disk

from common.data.modules import DataModule


class PreferenceDataModule(DataModule):
    def __init__(self, path: str, tokenizer, num_train_samples: int, num_test_samples: int,
                 batch_size: int = -1, random_seed: int = -1):
        self.path = path
        self.tokenizer = tokenizer
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.batch_size = batch_size

        self.random_seed = random_seed
        self.generator = torch.Generator().manual_seed(self.random_seed) if self.random_seed > 0 else None

    def setup(self):
        dataset = load_from_disk(self.path)
        self.train_dataset, self.test_dataset = dataset["train"], dataset["test"]

        self.num_train_samples = min(self.num_train_samples, len(self.train_dataset)) if self.num_train_samples > 0 else len(self.train_dataset)
        self.num_test_samples = min(self.num_test_samples, len(self.test_dataset)) if self.num_test_samples > 0 else len(self.test_dataset)

        if self.num_train_samples < len(self.train_dataset):
            perm = torch.randperm(len(self.train_dataset), generator=self.generator)
            self.train_dataset = self.train_dataset.select(perm[:self.num_train_samples])

        if self.num_test_samples < len(self.test_dataset):
            perm = torch.randperm(len(self.test_dataset), generator=self.generator)
            self.test_dataset = self.test_dataset.select(perm[:self.num_test_samples])

        self.prepared_train_dataloader = self.__create_train_dataloader()
        self.prepared_test_dataloader = self.__creat_test_dataloader()

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
