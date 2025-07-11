import argparse

from common.experiment.experiments_plan_runner import ExperimentsPlanRunner
from rm_exps.experiments.persona_controlled_reward_model_experiment import PersonaControlledRewardModelExperiment
from utils.arg_utils import parse_unknown_argparse_args_into_dict

def main():
    parser = argparse.ArgumentParser()
    ExperimentsPlanRunner.add_experiments_plan_runner_specific_args(parser)
    args, unknown_args = parser.parse_known_args()
    overrides_dict = parse_unknown_argparse_args_into_dict(unknown_args)

    experiments_plan_runner = ExperimentsPlanRunner()
    experiment = PersonaControlledRewardModelExperiment()
    experiments_plan_runner.run(plan_config_path=args.plan_config_path,
                                experiment=experiment,
                                disable_console_log=args.disable_console_log,
                                save_logs=args.save_logs,
                                log_dir=args.log_dir,
                                log_file_name_prefix=args.log_file_name_prefix,
                                overrides_dict=overrides_dict)


if __name__ == "__main__":
    main()
