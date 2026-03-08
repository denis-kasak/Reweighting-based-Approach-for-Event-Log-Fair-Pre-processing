from typing import Dict, List, Literal, Tuple

import pandas as pd


import pandas as pd
from typing import Dict, List, Literal, Tuple

def build_distribution_table(
    decision_table: pd.DataFrame,
    decision_points: Dict[str, List[str]],
    aggregation: Literal["existence", "count"] = "count",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    --------
    distribution_df :
        One row per (sensitive_attribute + decision vector) with:
        - group_id
        - P(sensitive_attribute, decisions)
        - decision columns

    group_case_mapping :
        DataFrame with columns:
        - case_id
        - group_id
    """

    valid_decisions = decision_table[
        decision_table["source"].notna() &
        decision_table["target"].notna()
    ]

    # All cases (including those with no decisions)
    all_cases = decision_table[["case_id", "sensitive_attribute"]].drop_duplicates()

    counts = (
        valid_decisions.groupby(
            ["case_id", "sensitive_attribute", "source", "target"]
        )
        .size()
        .reset_index(name="count")
    )

    counts["source_target"] = "#" + counts["source"] + "_" + counts["target"]


    case_level = counts.pivot_table(
        index=["case_id", "sensitive_attribute"],
        columns="source_target",
        values="count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()

    case_level = all_cases.merge(
        case_level,
        on=["case_id", "sensitive_attribute"],
        how="left",
    )

    expected_cols = [
        f"#{place}_{transition}"
        for place, transitions in decision_points.items()
        for transition in transitions
    ]

    for col in expected_cols:
        if col not in case_level.columns:
            case_level[col] = 0

    # Fill NaNs for no-decision cases with 0
    case_level[expected_cols] = case_level[expected_cols].fillna(0)

    case_level = case_level[["case_id", "sensitive_attribute"] + expected_cols]

    # Apply aggregation
    if aggregation == "existence":
        case_level[expected_cols] = (case_level[expected_cols] > 0).astype(int)
    elif aggregation != "count":
        raise ValueError("aggregation must be 'existence' or 'count'")

    # Build grouped distribution
    grouped = (
        case_level.groupby(["sensitive_attribute"] + expected_cols)
        .size()
        .reset_index(name="count")
    )

    total_cases = case_level["case_id"].nunique()
    grouped["P(sensitive_attribute, decisions)"] = grouped["count"] / total_cases

    grouped = grouped.sort_values(
        by="P(sensitive_attribute, decisions)",
        ascending=False,
    ).reset_index(drop=True)

    grouped["group_id"] = grouped.index + 1

    distribution_df = grouped[
        ["group_id", "P(sensitive_attribute, decisions)", "sensitive_attribute"]
        + expected_cols
    ]

    # Build group_id/case_id mapping (vectorized merge)
    group_case_mapping = case_level.merge(
        distribution_df[["group_id", "sensitive_attribute"] + expected_cols],
        on=["sensitive_attribute"] + expected_cols,
        how="left",
    )[["case_id", "group_id"]]

    return distribution_df, group_case_mapping

def choose_reference_group(distribution_table: pd.DataFrame):
    """
    Get the sensitive attribute that:
    - has rows that overall cover most unique place-transition pairs (most diverse decision patterns)
    - among those, has the highest overall probability mass in the distribution table
    """

    # Get total probability mass per sensitive attribute
    mass_per_attribute = (
        distribution_table.groupby("sensitive_attribute")[
            "P(sensitive_attribute, decisions)"
        ]
        .sum()
        .reset_index(name="total_mass")
    )

    # Get number of unique place-transition pairs per sensitive attribute
    decision_cols = [col for col in distribution_table.columns if col.startswith("#")]
    distribution_table["num_decisions"] = (
        distribution_table[decision_cols].gt(0).sum(axis=1)
    )
    diversity_per_attribute = (
        distribution_table.groupby("sensitive_attribute")["num_decisions"]
        .max()
        .reset_index(name="max_decisions")
    )

    # Merge mass and diversity info
    attribute_info = pd.merge(
        mass_per_attribute, diversity_per_attribute, on="sensitive_attribute"
    )

    # Choose attribute with highest diversity, then highest mass
    reference_group = attribute_info.sort_values(
        by=["max_decisions", "total_mass"], ascending=False
    ).iloc[0]["sensitive_attribute"]

    return reference_group
