import torch
import torch.nn as nn
from accelerate import Accelerator
from transformers import GenerationConfig
from trl.models.utils import unwrap_model_for_generation

from common.evaluation import metrics as metrics
from common.evaluation.evaluators import Evaluator, MetricsEvaluator
from rm_exps.data.preference_datamodule import PreferenceDataModule
from utils.data_utils import HamiltonianCycleDatasetLoader
from utils.rm_utils import convert_preference_batch_to_chat_format


class HamiltonianCycleGenerationEvaluator(Evaluator):
    TRAIN_HAM_CYCLE_GEN_ACC_METRIC_NAME = "eval train ham cycle gen accuracy"
    TEST_HAM_CYCLE_GEN_ACC_METRIC_NAME = "eval test ham cycle gen accuracy"
    MAX_TOKENS = 300

    def __init__(self, accelerator: Accelerator, model: nn.Module, tokenizer, datamodule: PreferenceDataModule, num_tries: int = 1, logger=None):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.datamodule = datamodule
        self.num_tries = num_tries
        self.logger = logger
        self.generation_config = GenerationConfig(
            max_new_tokens=self.MAX_TOKENS,
            temperature=1.0,
            top_k=0.0,
            top_p=1.0,
            do_sample=True
        )

        self.metric_infos = self.__create_metric_infos()
        self.metrics = {name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def __create_metric_infos(self):
        return {
            self.TRAIN_HAM_CYCLE_GEN_ACC_METRIC_NAME: metrics.MetricInfo(self.TRAIN_HAM_CYCLE_GEN_ACC_METRIC_NAME,
                                                                         metrics.DummyAveragedMetric(),
                                                                         tag="eval ham cycle gen accuracy"),
            self.TEST_HAM_CYCLE_GEN_ACC_METRIC_NAME: metrics.MetricInfo(self.TEST_HAM_CYCLE_GEN_ACC_METRIC_NAME,
                                                                        metrics.DummyAveragedMetric(),
                                                                        tag="eval ham cycle gen accuracy")
        }

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

    def __is_ham_cycle_gen_correct(self, prompt: str, generated_response: str):
        # Parse number of vertices from the prompt
        num_vertices = len(prompt.split("Vertices: ")[1].split("\n")[0].split(HamiltonianCycleDatasetLoader.SEP_TOKEN)[1:])

        # Parse edges from the prompt
        edges_str = prompt.split("Edges: ")[1].split(self.tokenizer.eos_token)[0]
        edges = [tuple(map(int, edge.split(HamiltonianCycleDatasetLoader.EDGE_SEP_TOKEN)))
                 for edge in edges_str.split(HamiltonianCycleDatasetLoader.SEP_TOKEN)[1:]]
        edges_set = set(edges)

        # Parse vertices from generation
        gen_vertices = generated_response.split(HamiltonianCycleDatasetLoader.SEP_TOKEN)
        if len(gen_vertices) != num_vertices or len(set(gen_vertices)) != num_vertices:
            return 0

        for i in range(len(gen_vertices)):
            if not gen_vertices[i].isdigit():
                return 0

            gen_vertices[i] = int(gen_vertices[i])

        for i in range(num_vertices):
            if (min(gen_vertices[i], gen_vertices[(i + 1) % num_vertices]),
                max(gen_vertices[i], gen_vertices[(i + 1) % num_vertices])) not in edges_set:
                return 0

        return 1

    def __compute_correct_gens_mat(self, prompts, per_prompt_generations):
        correct_gens_mat = []
        for i, prompt in enumerate(prompts):
            prompt_generations = [batch_generations[i] for batch_generations in per_prompt_generations]
            correct_gens_mat.append([self.__is_ham_cycle_gen_correct(prompt, gen) for gen in prompt_generations])

        return torch.tensor(correct_gens_mat, dtype=torch.float)

    def __compute_ham_cycle_gen_metrics(self, is_train: bool):
        dataloader = self.datamodule.train_dataloader() if is_train else self.datamodule.val_dataloader()

        logged_first_batch = False
        accs = []
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            for batch in dataloader:
                prompts = convert_preference_batch_to_chat_format(self.tokenizer, batch)["prompt"]
                tokenized_inputs = self.tokenizer(prompts, padding=True, truncation=False,
                                                  add_special_tokens=False, return_tensors="pt").to(self.accelerator.device)

                per_prompt_generations = []
                for _ in range(self.num_tries):
                    outputs = self.model.generate(**tokenized_inputs, generation_config=self.generation_config)
                    decoded_generations = self.tokenizer.batch_decode(outputs[:, tokenized_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    per_prompt_generations.append(decoded_generations)

                correct_gens_mat = self.__compute_correct_gens_mat(prompts, per_prompt_generations)
                accs.append((correct_gens_mat.sum(dim=1) > 0).float())

                if not logged_first_batch:
                    prompts_str = prompts[0]
                    generations_for_prompt = [per_prompt_gen[0] for per_prompt_gen in per_prompt_generations]
                    self.logger.info(f"Hamiltonian cycle generation evaluation: batch with {len(prompts)} prompts, {self.num_tries} tries")
                    self.logger.info(f"Example prompt: {prompts_str}\n"
                                     f"Example generated responses: {generations_for_prompt}\n"
                                     f"Correct generations accuracy in batch: {(correct_gens_mat.sum(dim=1) > 0).float().mean().item()}")
                    logged_first_batch = True

        acc = torch.cat(accs).mean().item()
        ham_cycle_gen_acc_metric_name = self.TRAIN_HAM_CYCLE_GEN_ACC_METRIC_NAME if is_train else self.TEST_HAM_CYCLE_GEN_ACC_METRIC_NAME
        self.__populate_metric_value(ham_cycle_gen_acc_metric_name, acc)

    def evaluate(self):
        with torch.no_grad():
            self.__compute_ham_cycle_gen_metrics(is_train=True)
            self.__compute_ham_cycle_gen_metrics(is_train=False)
            eval_metric_values = {name: metric.current_value() for name, metric in self.metrics.items()}
            return eval_metric_values
