from typing import Dict, List, Tuple, Union

import pandas as pd
from pm4py.objects.log.obj import EventLog
from tqdm import tqdm


def get_decision_points_dfg(directly_follows_pairs: Dict) -> Dict[str, List[str]]:
    """
    Identifies decision points in the DFG (activities with >=2 outgoing arcs).
    """

    decisions = []

    for (source, target), count in directly_follows_pairs.items():
        decisions.append((source, target))

    dp = pd.DataFrame(decisions, columns=["source", "target"])

    # Keep only sources with >= 2 outgoing transitions
    dp = dp.groupby("source").filter(lambda x: len(x) >= 2)

    # Convert to dict {source -> [targets]}
    decision_points = dp.groupby("source")["target"].apply(list).to_dict()
    
    return decision_points


def filter_relevant_decisions(
    decision_points: Dict[str, List[str]], relevant_decisions: List[Tuple[str, str]]
) -> Dict[str, List[str]]:
    """Filter decision points to keep only those matching relevant_decisions.

    Each relevant decision is a (source, target) tuple where None acts as a wildcard.
    """
    if not relevant_decisions:
        return decision_points

    filtered_decision_points = {}

    for place, transitions in decision_points.items():
        filtered_transitions = []
        for transition in transitions:
            for rel_source, rel_target in relevant_decisions:
                if (rel_source is None or place == rel_source) and (
                    rel_target is None or transition == rel_target
                ):
                    filtered_transitions.append(transition)
                    break
        if filtered_transitions:
            filtered_decision_points[place] = filtered_transitions

    return filtered_decision_points


def build_decision_table_dfg(
    log: Union[EventLog, pd.DataFrame],
    decision_points: Dict[str, List[str]],
    sensitive_attribute: str,
) -> pd.DataFrame:
    """
    Build a single global decision table containing all decisions for all activities in the DFG.

    The returned DataFrame has columns:
    - 'source': the decision point activity name
    - 'target': the chosen next activity name
    - One column per attribute (both event and trace attributes
    """

    decisions = []

    for trace in log:
        case_id = trace.attributes["concept:name"]
        sensitive_attribute_val = _get_sensitive_attribute_value(
            trace, sensitive_attribute
        )

        has_decision = False
        for i in range(len(trace)):
            source_activity = trace[i]["concept:name"]
            if source_activity in decision_points.keys():
                has_decision = True
                target_activity = (
                    trace[i + 1]["concept:name"] if i + 1 < len(trace) else None
                )
                decisions.append(
                    {
                        "source": source_activity,
                        "target": target_activity,
                        "case_id": case_id,
                        "sensitive_attribute": sensitive_attribute_val,
                    }
                )
        if not has_decision:
            # Add a row with NaN target to indicate no decision point in this trace
            decisions.append(
                {
                    "source": None,
                    "target": None,
                    "case_id": case_id,
                    "sensitive_attribute": sensitive_attribute_val,
                }
            )

    if not decisions:
        return pd.DataFrame()
    else:
        return pd.DataFrame(decisions)


def _get_sensitive_attribute_value(trace, sensitive_attribute):
    # First check trace attributes
    if sensitive_attribute in trace.attributes:
        return trace.attributes[sensitive_attribute]
    # If not found, check event attributes in order
    for event in trace:
        if sensitive_attribute in event:
            return event[sensitive_attribute]
    return None
