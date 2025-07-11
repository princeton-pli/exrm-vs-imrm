from abc import ABC, abstractmethod
from typing import List, Tuple, Set

import torch
from datasets import load_dataset, Dataset


class PreferenceDatasetLoader(ABC):

    @abstractmethod
    def load_and_preprocess_preference_dataset(self, cache_dir: str = None, rnd_gen: torch.Generator = None, **kwargs):
        """
        Returns a tuple of (train_dataset, test_dataset), where the format of examples in each dataset should be "prompt", "chosen", "rejected".
        The "chosen" and "rejected" responses need to include only the text of the response, not the prompt.
        """
        raise NotImplementedError


class UltraFeedbackDatasetLoader(PreferenceDatasetLoader):

    def __prepare_example_format(self, example: dict):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"][1]["content"],
            "rejected": example["rejected"][1]["content"]
        }

    def load_and_preprocess_preference_dataset(self, cache_dir: str = None, rnd_gen: torch.Generator = None, **kwargs):
        dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", cache_dir=cache_dir)
        train_dataset = dataset["train_prefs"]
        test_dataset = dataset["test_prefs"]

        # Filter out samples whose chosen and rejected response scores are equal
        train_dataset = train_dataset.filter(lambda example: example["score_chosen"] > example["score_rejected"])
        test_dataset = test_dataset.filter(lambda example: example["score_chosen"] > example["score_rejected"])

        train_dataset = train_dataset.map(self.__prepare_example_format, batched=False)
        train_dataset = train_dataset.select_columns(["prompt", "chosen", "rejected"])
        test_dataset = test_dataset.map(self.__prepare_example_format, batched=False)
        test_dataset = test_dataset.select_columns(["prompt", "chosen", "rejected"])

        return train_dataset, test_dataset


class RewardBenchPRMMathDatasetLoader(PreferenceDatasetLoader):
    NUM_TEST_SAMPLES = 400

    def load_and_preprocess_preference_dataset(self, cache_dir: str = None, rnd_gen: torch.Generator = None, **kwargs):
        dataset = load_dataset("allenai/reward-bench", cache_dir=cache_dir)["filtered"]
        dataset = dataset.filter(lambda example: example["subset"] == "math-prm")
        dataset = dataset.select_columns(["prompt", "chosen", "rejected"])

        perm = torch.randperm(len(dataset), generator=rnd_gen)
        train_dataset = dataset.select(perm[:len(dataset) - self.NUM_TEST_SAMPLES])
        test_dataset = dataset.select(perm[len(dataset) - self.NUM_TEST_SAMPLES:])
        return train_dataset, test_dataset


class RewardBenchCodeDatasetLoader(PreferenceDatasetLoader):
    NUM_TEST_SAMPLES = 492

    def load_and_preprocess_preference_dataset(self, cache_dir: str = None, rnd_gen: torch.Generator = None, **kwargs):
        dataset = load_dataset("allenai/reward-bench", cache_dir=cache_dir)["filtered"]
        dataset = dataset.filter(lambda example: example["subset"].startswith("hep"))
        dataset = dataset.select_columns(["prompt", "chosen", "rejected"])

        perm = torch.randperm(len(dataset), generator=rnd_gen)
        train_dataset = dataset.select(perm[:len(dataset) - self.NUM_TEST_SAMPLES])
        test_dataset = dataset.select(perm[len(dataset) - self.NUM_TEST_SAMPLES:])
        return train_dataset, test_dataset


class RewardMATHDatasetLoader(PreferenceDatasetLoader):
    NUM_TEST_SAMPLES = 1000

    def __prepare_example_format(self, example: dict):
        chosen_step_list = example["eval_solution_A"] if example["chosen_position"] == "A" else example["eval_solution_B"]
        rejected_step_list = example["eval_solution_B"] if example["chosen_position"] == "A" else example["eval_solution_A"]
        return {
            "prompt": example["problem"],
            "chosen": " ".join(chosen_step_list),
            "rejected": " ".join(rejected_step_list)
        }

    def load_and_preprocess_preference_dataset(self, cache_dir: str = None, rnd_gen: torch.Generator = None, **kwargs):
        dataset = load_dataset("RewardMATH/RewardMATH_pairwise", cache_dir=cache_dir)["test"]
        dataset = dataset.map(self.__prepare_example_format, batched=False)
        dataset = dataset.select_columns(["prompt", "chosen", "rejected"])

        perm = torch.randperm(len(dataset), generator=rnd_gen)
        train_dataset = dataset.select(perm[:len(dataset) - self.NUM_TEST_SAMPLES])
        test_dataset = dataset.select(perm[len(dataset) - self.NUM_TEST_SAMPLES:])
        return train_dataset, test_dataset


class HamiltonianCycleDatasetLoader(PreferenceDatasetLoader):
    # Graphs with Hamiltonian cycles are generated following Section IV.B of https://arxiv.org/pdf/2306.06523:
    # 1. Randomly sample a permutation of vertices and add edges that create a Hamiltonian cycle with these vertices.
    # 2. For each other edge, add it with probability p (i.i.d.).

    NUM_TRAIN_SAMPLES = 5000
    NUM_TEST_SAMPLES = 200
    MAX_ATTEMPTS_NOT_HAM_CYCLE = 10
    SEP_TOKEN = "<sep>"
    EDGE_SEP_TOKEN = "<edge_sep>"

    def __generate_rnd_graph_with_hamiltonian_cycle(self, num_vertices: int, p: float, rnd_gen: torch.Generator = None):
        vertices = [i for i in range(num_vertices)]
        ham_cycle = torch.randperm(num_vertices, generator=rnd_gen).tolist()
        edges = [(min(ham_cycle[i], ham_cycle[(i + 1) % num_vertices]), max(ham_cycle[i], ham_cycle[(i + 1) % num_vertices]))
                 for i in range(num_vertices)]

        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                if (i, j) in edges:
                    continue

                if torch.rand(1, generator=rnd_gen).item() < p:
                    edges.append((i, j))

        shuffled_indices = torch.randperm(len(edges), generator=rnd_gen)
        edges = [edges[i] for i in shuffled_indices]
        edges_set = set(edges)
        not_ham_cycle = self.__get_vertices_perm_that_is_not_a_ham_cycle(vertices, edges_set, rnd_gen)

        return vertices, edges, ham_cycle, not_ham_cycle

    def __get_vertices_perm_that_is_not_a_ham_cycle(self, vertices: List[int], edges_set: Set[Tuple[int, int]], rnd_gen: torch.Generator = None):
        for _ in range(self.MAX_ATTEMPTS_NOT_HAM_CYCLE):
            perm = torch.randperm(len(vertices), generator=rnd_gen).tolist()

            for i in range(len(perm)):
                if (min(perm[i], perm[(i + 1) % len(perm)]), max(perm[i], perm[(i + 1) % len(perm)])) not in edges_set:
                    return perm

        raise ValueError("Failed to generate a permutation that is not a Hamiltonian cycle.")

    def __convert_graph_to_example_format(self, vertices: List[int], edges: List[Tuple[int, int]], ham_cycle: List[int],
                                          not_ham_cycle: List[int]) -> dict:
        vertices_str = f"Vertices: {self.SEP_TOKEN}" + self.SEP_TOKEN.join(map(str, vertices))
        edges_str = f"Edges: {self.SEP_TOKEN}" + self.SEP_TOKEN.join([f"{u}{self.EDGE_SEP_TOKEN}{v}" for u, v in edges])
        return {
            "prompt": f"{vertices_str}\n{edges_str}",
            "chosen": self.SEP_TOKEN.join(map(str, ham_cycle)),
            "rejected": self.SEP_TOKEN.join(map(str, not_ham_cycle))
        }

    def __create_dataset_of_formatted_examples(self, num_samples: int, num_vertices: int, edge_p: float, rnd_gen: torch.Generator = None):
        examples = []
        for _ in range(num_samples):
            vertices, edges, ham_cycle, not_ham_cycle = self.__generate_rnd_graph_with_hamiltonian_cycle(num_vertices, edge_p, rnd_gen)
            examples.append(self.__convert_graph_to_example_format(vertices, edges, ham_cycle, not_ham_cycle))

        return Dataset.from_list(examples)

    def load_and_preprocess_preference_dataset(self, cache_dir: str = None, rnd_gen: torch.Generator = None, num_vertices: int = 10,
                                               edge_p: float = 0.05, **kwargs):
        return (self.__create_dataset_of_formatted_examples(self.NUM_TRAIN_SAMPLES, num_vertices, edge_p, rnd_gen),
                self.__create_dataset_of_formatted_examples(self.NUM_TEST_SAMPLES, num_vertices, edge_p, rnd_gen))


def get_preference_dataset_loader(dataset: str):
    if dataset == "ultrafeedback":
        return UltraFeedbackDatasetLoader()
    elif dataset == "rewardbench_math":
        return RewardBenchPRMMathDatasetLoader()
    elif dataset == "rewardbench_code":
        return RewardBenchCodeDatasetLoader()
    elif dataset == "rewardmath":
        return RewardMATHDatasetLoader()
    elif dataset == "hamiltonian_cycle":
        return HamiltonianCycleDatasetLoader()
    else:
        raise ValueError(f"Unsupported: {dataset}")
