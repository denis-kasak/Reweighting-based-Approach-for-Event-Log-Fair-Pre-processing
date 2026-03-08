import graphviz
import pandas as pd
from numpy import max


def visualize_dfg_independence(
    directly_follows_pairs,
    start_activities,
    end_activities,
    independence_results: pd.DataFrame,
    filename: str,
    target_ind: float,
) -> None:
    dot = graphviz.Digraph(format="png")

    # Add start and end nodes
    dot.node("start", shape="circle", style="filled", fillcolor="lightblue")
    dot.node("end", shape="circle", style="filled", fillcolor="lightblue")

    # Add directly follows pairs with color coding based on IND
    for source, target in directly_follows_pairs:
        ind_values = independence_results[
            (independence_results["source"] == source)
            & (independence_results["target"] == target)
        ]["IND"].values
        if ind_values.size > 0:

            # Drop NaN values before calculating mean
            ind_values = ind_values[~pd.isna(ind_values)]

            label = ""

            for _, row in independence_results[
                (independence_results["source"] == source)
                & (independence_results["target"] == target)
            ].iterrows():
                if row["IND"] >= target_ind:
                    label += f"{row['sensitive_attribute']}: {row['IND']:.2f}\n"

            if ind_values.size > 0:
                ind_value = max(ind_values)

                if ind_value <= target_ind:
                    color = "green"
                else:
                    color = "red"

                dot.edge(source, target, label=label, color=color)
            else:
                dot.edge(source, target)
        else:
            dot.edge(source, target)

    # Add edges from start to start activities and from end activities to end
    for start_act in start_activities:
        dot.edge("start", start_act)
    for end_act in end_activities:
        dot.edge(end_act, "end")

    # Render the graph    dot.render(filename, view=False)
    dot.render(filename, view=False)


def save_ind_stats(
    independence_results: pd.DataFrame, target_ind: float, file_path: str
) -> None:

    #### Number of decisions where at least one group has IND above target_ind

    # group by source and target, then check if any IND value in the group is above target_ind
    decisions_above_target = independence_results.groupby(["source", "target"]).apply(
        lambda group: (group["IND"] >= target_ind).any()
    )

    num_decisions_above_target = decisions_above_target.sum()
    total_decisions = len(decisions_above_target)
    percentage_above_target = (num_decisions_above_target / total_decisions) * 100

    with open(file_path, "w") as f:
        f.write(
            f"Number of decisions with IND above target ({target_ind}): {num_decisions_above_target}\n"
        )
        f.write(f"Total number of decisions: {total_decisions}\n")
        f.write(
            f"Percentage of decisions above target: {percentage_above_target:.2f}%\n"
        )
