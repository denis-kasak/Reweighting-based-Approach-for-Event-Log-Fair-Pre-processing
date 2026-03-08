import json
import os
from argparse import ArgumentParser
from typing import Dict, List, Literal, Tuple
import pandas as pd
from pm4py import (
    discover_dfg,
)
from pm4py.objects.log.importer.xes.importer import apply as xes_importer
from pm4py.objects.log.exporter.xes.exporter import apply as xes_exporter
from pm4py.objects.log.obj import EventLog

from decisions import (
    build_decision_table_dfg,
    filter_relevant_decisions,
    get_decision_points_dfg,
)
from distribution import build_distribution_table, choose_reference_group
from optimizer import (
    TreeStructuredParzenEstimator,
    calculate_decision_rates,
    calculate_independence,
)
from utils import convert_bool_to_int
from visualize import (
    save_ind_stats,
    visualize_dfg_independence,
)


def get_results(
    distribution_table: pd.DataFrame,
    dp: Dict[str, List[str]],
    reference_group: str,
    parent_output_dir: str,
):
    decision_rates: pd.DataFrame = calculate_decision_rates(
        distribution_table=distribution_table, decision_points=dp
    )
    decision_rates.to_csv(f"{parent_output_dir}/base_decision_rates.csv", index=False)
    independence: pd.DataFrame = calculate_independence(
        decision_rates=decision_rates, reference_group=reference_group
    )
    independence.to_csv(f"{parent_output_dir}/base_independence.csv", index=False)

    return independence


def get_reweighed_results(
    log: EventLog,
    group_case_mapping,
    eventlog_name,
    distribution_table: pd.DataFrame,
    dp: Dict[str, List[str]],
    reference_group: str,
    parent_output_dir: str,
    num_trials: int,
    n_repetitions: int,
    target_ind: float,
    optimizer_dp: Dict[str, List[str]],
    export_log: bool,
    optimizer_ind_aggregation: str,
    optimizer_seed: int,
):
    optimizer = TreeStructuredParzenEstimator()

    reweighed_distribution_table = optimizer.optimize_weights(
        distribution_table=distribution_table,
        reference_group=reference_group,
        parent_output_dir=parent_output_dir,
        n_trials=num_trials,
        n_repetitions=n_repetitions,
        target_ind=target_ind,
        optimizer_dp=optimizer_dp,
        optimizer_ind_aggregation=optimizer_ind_aggregation,
        optimizer_seed=optimizer_seed,
    )
    # export
    reweighed_distribution_table.to_csv(
        f"{parent_output_dir}/reweighed_distribution_table.csv", index=False
    )

    reweighed_decision_rates = calculate_decision_rates(
        distribution_table=reweighed_distribution_table,
        decision_points=dp,
    )
    reweighed_independence = calculate_independence(
        decision_rates=reweighed_decision_rates, reference_group=reference_group
    )
    reweighed_independence.to_csv(
        f"{parent_output_dir}/reweighed_independence.csv", index=False
    )

    reweighed_log = add_weights_as_attribute(
        log=log,
        distribution_table=reweighed_distribution_table,
        group_case_mapping=group_case_mapping,
    )

    print("Exporting reweighed log to XES format...")

    if export_log:
        xes_exporter(
            reweighed_log, f"{parent_output_dir}/{eventlog_name}_reweighed.xes"
        )

    return reweighed_independence, reweighed_log


def add_weights_as_attribute(
    log: EventLog,
    distribution_table: pd.DataFrame,
    group_case_mapping: pd.DataFrame,
) -> EventLog:

    # --- Precompute mappings (O(n)) ---

    # case_id -> group_id
    case_to_group = dict(
        zip(group_case_mapping["case_id"], group_case_mapping["group_id"])
    )

    # group_id -> number of cases in group
    group_sizes = group_case_mapping["group_id"].value_counts().to_dict()

    # group_id -> probability
    group_probs = dict(
        zip(
            distribution_table["group_id"],
            distribution_table["P(sensitive_attribute, decisions)"],
        )
    )

    # --- Assign weights (O(n)) ---
    total_weight = 0.0

    for trace in log:
        case_id = trace.attributes["concept:name"]
        group = case_to_group[case_id]

        weight = group_probs[group] / group_sizes[group]
        trace.attributes["fairness:weight"] = weight

        total_weight += weight

    print(f"Total weight in log: {total_weight}")

    return log


def full_experiment(optimizer_seed: int):

    main_output_dir = f"./full_experiment_output_{optimizer_seed}"

    os.makedirs(main_output_dir, exist_ok=True)

    def folder_with_config_exists(config):
        config_filename = f"config.json"
        for folder_name in os.listdir(main_output_dir):
            folder_path = os.path.join(main_output_dir, folder_name)
            if os.path.isdir(folder_path):
                config_path = os.path.join(folder_path, config_filename)
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        existing_config = json.load(f)
                        if (
                            existing_config["run_experiment_id"]
                            == config["run_experiment_id"]
                        ):
                            return True
        return False

    EVENTLOGS = [
        {
            "name": "hb_-age_+gender",
            "eventlog_path": "./eventlogs/hb_-age_+gender.xes",
            "sensitive_attribute": "gender",
            "user-defined_decisions": [
                ("RELEASE", "CODE OK"),
                ("RELEASE", "CODE NOK"),
            ],
        },
        {
            "name": "bpi_2012",
            "eventlog_path": "./eventlogs/bpi_2012.xes",
            "sensitive_attribute": "gender",
            "user-defined_decisions": [
                ("A_PARTLYSUBMITTED", "A_DECLINED"),
                ("A_PARTLYSUBMITTED", "A_PREACCEPTED"),
            ],
        },
        {
            "name": "cs",
            "eventlog_path": "./eventlogs/cs.xes",
            "sensitive_attribute": "gender",
            "user-defined_decisions": [
                ("asses eligibility", "collect history"),
                ("asses eligibility", "refuse screening"),
            ],
        },
    ]

    NUM_TRIALS = [10, 50, 100, 250]
    OPTMIZER_TARGET_DECISIONS = ["all", "user-defined"]
    N_REPETITIONS = 1

    all_possible_configs = []

    run_experiment_id = 1
    for eventlog in EVENTLOGS:
        for optimizer_target_decisions in OPTMIZER_TARGET_DECISIONS:
            for num_trial in NUM_TRIALS:
                config = {
                    "run_experiment_id": run_experiment_id,
                    "eventlog": eventlog["name"],
                    "eventlog_path": eventlog["eventlog_path"],
                    "sensitive_attribute": eventlog["sensitive_attribute"],
                    "user-defined_decisions": eventlog["user-defined_decisions"],
                    "group_aggregation": "count",
                    "optimizer_target_ind": 0.0,
                    "optimizer_target_decisions": optimizer_target_decisions,
                    "num_trials": num_trial,
                    "optimizer_ind_aggregation": "mean",
                }
                all_possible_configs.append(config)
                run_experiment_id += 1

    for config in all_possible_configs:
        if folder_with_config_exists(config):
            print(f"Experiment with config {config} already exists. Skipping...")
            continue

        output_dir = f"{main_output_dir}/run_experiment_{config['run_experiment_id']}"
        os.makedirs(output_dir, exist_ok=True)

        log = xes_importer(config["eventlog_path"])
        log = convert_bool_to_int(log, config["sensitive_attribute"])
        directly_follows_pairs, start_activities, end_activities = discover_dfg(log)
        dp = get_decision_points_dfg(directly_follows_pairs)
        dt = build_decision_table_dfg(log, dp, config["sensitive_attribute"])
        distribution_table, group_case_mapping = build_distribution_table(
            dt, dp, aggregation=config["group_aggregation"]
        )
        reference_group = choose_reference_group(distribution_table)
        base_independence = get_results(
            distribution_table=distribution_table,
            dp=dp,
            reference_group=reference_group,
            parent_output_dir=output_dir,
        )
        visualize_dfg_independence(
            directly_follows_pairs,
            start_activities,
            end_activities,
            base_independence,
            filename=f"{output_dir}/base_dfg_independence",
            target_ind=0.2,
        )
        save_ind_stats(
            base_independence,
            target_ind=config["optimizer_target_ind"],
            file_path=f"{output_dir}/base_ind_stats.txt",
        )

        if config["optimizer_target_decisions"] == "all":
            optimizer_dp = dp.copy()
        elif config["optimizer_target_decisions"] == "user-defined":
            optimizer_dp = filter_relevant_decisions(
                dp, config["user-defined_decisions"]
            )
        reweighed_independence, reweighed_log = get_reweighed_results(
            log=log,
            group_case_mapping=group_case_mapping,
            eventlog_name=config["eventlog"],
            distribution_table=distribution_table,
            dp=dp,
            reference_group=reference_group,
            parent_output_dir=output_dir,
            num_trials=config["num_trials"],
            n_repetitions=N_REPETITIONS,
            target_ind=config["optimizer_target_ind"],
            optimizer_dp=optimizer_dp,
            optimizer_ind_aggregation=config["optimizer_ind_aggregation"],
            export_log=True,
            optimizer_seed=optimizer_seed,
        )
        visualize_dfg_independence(
            directly_follows_pairs,
            start_activities,
            end_activities,
            reweighed_independence,
            filename=f"{output_dir}/reweighed_dfg_independence",
            target_ind=0.2,
        )
        save_ind_stats(
            reweighed_independence,
            target_ind=config["optimizer_target_ind"],
            file_path=f"{output_dir}/reweighed_ind_stats.txt",
        )
        with open(f"{output_dir}/config.json", "w") as f:
            json.dump(config, f, indent=4)


if __name__ == "__main__":
    for optimizer_seed in [1, 2, 3]:
        full_experiment(optimizer_seed=optimizer_seed)
