from typing import Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from numpy import mean
from optuna import create_study
from optuna.samplers import TPESampler
from optuna.trial import Trial
from scipy.special import softmax


def calculate_decision_rates(
    distribution_table: pd.DataFrame, decision_points: Dict[str, List[str]]
) -> pd.DataFrame:
    # Calculate decision rates for each place-transition-sens_attr combination

    decision_rates_aggregated = []

    decision_rates = []

    # Go over each row in distribution table
    for _, row in distribution_table.iterrows():

        sensitive_attribute = row["sensitive_attribute"]

        for source in decision_points.keys():
            relevant_columns = [
                col
                for col in distribution_table.columns
                if col.startswith(f"#{source}_")
            ]

            place_occurrence = row[relevant_columns].sum()

            for target in decision_points[source]:
                relevant_column = f"#{source}_{target}"

                transition_occurrence = row[relevant_column]

                rate = (
                    transition_occurrence / place_occurrence
                    if place_occurrence > 0
                    else None
                )

                decision_rates.append(
                    {
                        "source": source,
                        "target": target,
                        "sensitive_attribute": sensitive_attribute,
                        "rate": rate,
                        "weight": row["P(sensitive_attribute, decisions)"],
                    }
                )

    decision_rates_df = pd.DataFrame(decision_rates)

    # group by source, target, sensitive_attribute and calculate weighted average of rates
    decision_rates_aggregated = (
        decision_rates_df.groupby(["source", "target", "sensitive_attribute"])
        .apply(
            lambda x: (
                (x["rate"] * x["weight"]).sum() / x["weight"].sum()
                if x["weight"].sum() > 0
                else None
            )
        )
        .reset_index(name="rate")
    )

    return decision_rates_aggregated


def calculate_independence(
    decision_rates: pd.DataFrame, reference_group
) -> pd.DataFrame:
    # IND = | rate_reference_group - rate_other_group |
    independence_results = []

    reference_rates = decision_rates[
        decision_rates["sensitive_attribute"] == reference_group
    ].set_index(["source", "target"])["rate"]

    for _, row in decision_rates.iterrows():
        if row["sensitive_attribute"] == reference_group:
            continue

        ref_rate = reference_rates.get((row["source"], row["target"]), 0.0)
        ind = abs(row["rate"] - ref_rate)

        independence_results.append(
            {
                "source": row["source"],
                "target": row["target"],
                "sensitive_attribute": row["sensitive_attribute"],
                "IND": ind,
            }
        )

    return pd.DataFrame(independence_results)


class TreeStructuredParzenEstimator:

    def __init__(self):
        super().__init__()

    def _plot_optimization_history(self, trials: List[Trial], file_path: str):
        # Line plot of IND values over trials
        ind_values = [trial.values[0] for trial in trials]
        trial_numbers = list(range(1, len(trials) + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(trial_numbers, ind_values, color="blue", marker="o")
        plt.title("Optimization History")
        plt.xlabel("Trial Number")
        plt.ylabel("Mean IND")
        plt.grid()
        plt.savefig(file_path)
        plt.close()

        # Line plot of best IND value over trials
        best_ind_values = [
            min(trials[: i + 1], key=lambda t: t.values[0]).values[0]
            for i in range(len(trials))
        ]
        plt.figure(figsize=(10, 6))
        plt.plot(trial_numbers, best_ind_values, color="red", marker="o")
        plt.title("Best IND Value Over Trials")
        plt.xlabel("Trial Number")
        plt.ylabel("Best Mean IND")
        plt.grid()
        plt.savefig(file_path.replace(".png", "_best.png"))
        plt.close()

    def _plot_weight_distributions(
        self,
        original_distribution_table: pd.DataFrame,
        adjusted_distribution_table: pd.DataFrame,
        file_path: str,
    ):
        merged = pd.merge(
            original_distribution_table,
            adjusted_distribution_table,
            on=["group_id"],
            suffixes=("_original", "_adjusted"),
        )

        plt.figure(figsize=(10, 6))
        plt.scatter(
            merged["P(sensitive_attribute, decisions)_original"],
            merged["P(sensitive_attribute, decisions)_adjusted"],
            color="blue",
            edgecolors="black",
            linewidths=1.5,
            s=100,
        )
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.title("Original vs Adjusted Weights")
        plt.xlabel("Original Weight")
        plt.ylabel("Adjusted Weight")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid()
        plt.savefig(file_path)
        plt.close()

    def _apply_softmax_weights(
        self,
        distribution_table: pd.DataFrame,
    ) -> pd.DataFrame:
        group_weights = (
            distribution_table[["group_id", "P(sensitive_attribute, decisions)"]]
            .drop_duplicates()
            .set_index("group_id")["P(sensitive_attribute, decisions)"]
        )

        # standardize weights before softmax: mean=0 and std=1
        group_weights = (group_weights - group_weights.mean()) / group_weights.std()
        normalized_weights = softmax(group_weights)

        weight_mapping = dict(zip(group_weights.index, normalized_weights))
        distribution_table["P(sensitive_attribute, decisions)"] = distribution_table[
            "group_id"
        ].map(weight_mapping)
        return distribution_table

    def _apply_best_weights(
        self,
        distribution_table: pd.DataFrame,
        best_params: dict,
    ) -> pd.DataFrame:

        distribution_table = distribution_table.copy()

        for param in best_params:
            group_id = int(param.replace("weight_", ""))
            weight = best_params[param]
            distribution_table.loc[
                distribution_table["group_id"] == group_id,
                "P(sensitive_attribute, decisions)",
            ] = weight
        distribution_table = self._apply_softmax_weights(distribution_table)
        return distribution_table

    def optimize_weights(
        self,
        distribution_table: pd.DataFrame,
        reference_group,
        parent_output_dir: str,
        n_trials: int,
        n_repetitions: int,
        target_ind: float,
        optimizer_dp: Dict[str, List[str]],
        optimizer_ind_aggregation: Literal["mean", "max"],
        optimizer_seed: int,
    ) -> pd.DataFrame:

        def objective(trial: Trial):

            dt = distribution_table.copy()

            # Assign "weight" to each dt row based on "group_id"
            dt["P(sensitive_attribute, decisions)"] = dt["group_id"].apply(
                lambda group_id: trial.suggest_float(
                    f"weight_{group_id}", low=-10.0, high=10.0
                )
            )

            # Apply *softmax* to group weights
            dt = self._apply_softmax_weights(dt)

            decision_rates = calculate_decision_rates(
                distribution_table=dt, decision_points=optimizer_dp
            )

            independence = calculate_independence(
                decision_rates=decision_rates, reference_group=reference_group
            )

            # exclude reference group from optimization metric
            independence = independence[
                independence["sensitive_attribute"] != reference_group
            ]

            # substract target_ind from independence and cap at 0
            independence["IND"] = independence["IND"].apply(
                lambda x: max(x - target_ind, 0)
            )
            independence["IND"] = independence["IND"] / (1 - target_ind)

            # exclude rows with NaN IND values (due to 0/0 rates)
            independence = independence.dropna(subset=["IND"])

            ind_aggregation = mean if optimizer_ind_aggregation == "mean" else max

            return ind_aggregation(independence["IND"])

        best_in_repetitions = []

        for i_repetition in range(n_repetitions):

            study = create_study(
                direction="minimize",
                sampler=TPESampler(seed=optimizer_seed + i_repetition),
            )

            study.optimize(objective, n_trials=n_trials)

            self._plot_optimization_history(
                study.trials,
                file_path=f"{parent_output_dir}/optimization_history_rep_{i_repetition}.png",
            )

            best_trial = study.best_trial
            best_params = best_trial.params
            best_values = best_trial.values

            reweighed_distribution_table = self._apply_best_weights(
                distribution_table=distribution_table,
                best_params=best_params,
            )

            best_in_repetitions.append(
                {
                    "repetition": i_repetition,
                    "seed": i_repetition,
                    "best_params": best_params,
                    "IND": best_values[0],
                }
            )

        best_overall = min(best_in_repetitions, key=lambda res: res["IND"])

        best_overall_params = best_overall["best_params"]
        reweighed_distribution_table = self._apply_best_weights(
            distribution_table=distribution_table,
            best_params=best_overall_params,
        )

        # Save repetition results to csv:
        pd.DataFrame(best_in_repetitions).drop(columns=["best_params"]).to_csv(
            f"{parent_output_dir}/best_in_repetitions.csv", index=False
        )

        # Save min, max, mean, std of best values across repetitions to txt file:
        min_ind = min([res["IND"] for res in best_in_repetitions])
        max_ind = max([res["IND"] for res in best_in_repetitions])
        mean_ind = mean([res["IND"] for res in best_in_repetitions])
        std_ind = pd.DataFrame(best_in_repetitions)["IND"].std()

        with open(f"{parent_output_dir}/summary.txt", "w") as f:
            f.write(
                f"IND - Min: {min_ind}, Max: {max_ind}, Mean: {mean_ind}, Std: {std_ind}\n"
            )

        self._plot_weight_distributions(
            original_distribution_table=distribution_table,
            adjusted_distribution_table=reweighed_distribution_table,
            file_path=f"{parent_output_dir}/weight_distributions.png",
        )

        return reweighed_distribution_table
