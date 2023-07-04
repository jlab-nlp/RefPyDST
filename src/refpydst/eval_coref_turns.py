import logging
import os
import sys
from typing import List, Tuple, Callable, Dict, Set

import numpy as np
import wandb

from refpydst.db.ontology import normalize

from refpydst.artifacts import get_running_logs_by_group, read_run_artifact_logs
from refpydst.data.multiwoz23 import get_coreference_annotations
from refpydst.data_types import Turn, SlotName, SlotValue
from refpydst.error_analysis import evaluate_logs
from utils.general import WANDB_ENTITY, WANDB_PROJECT


def eval_on_given_turns(runs: List[List[Turn]]) -> Tuple[float, float]:
    jgas: List[float] = []
    for logs in runs:
        logs = evaluate_logs(logs)
        jgas.append(np.mean([t['jga'] for t in logs]).item())
    return np.mean(jgas).item(), np.std(jgas).item()


def eval_just_coreference_slots(runs: List[List[Turn]], coreferences) -> Tuple[float, float, float]:
    accs: List[float] = []
    total_coref_slots: int = 0
    for logs in runs:
        n_correct, n_total = 0, 0
        for i, log in enumerate(logs):
            dial_id, turn_id = log['ID'], log['turn_id']
            if dial_id in coreferences and turn_id in coreferences[dial_id]:
                coreferred_slots: Dict[SlotName, Tuple[SlotValue, str]] = coreferences[dial_id][turn_id]
                for slot_name, co_ref_dict in coreferred_slots.items():
                    if slot_name not in log['slot_values']:
                        logging.info(f"coreference annotation on {dial_id}-{turn_id} is not in gold state, "
                                     f"possible annotation correction")
                        continue
                    n_total += 1
                    # if we have a prediction for the slot, check correctness, otherwise wrong
                    if slot_name in log['pred']:
                        predicted_value: SlotValue = log['pred'][slot_name]
                        gold_value = log['slot_values'][slot_name]
                        if predicted_value == gold_value:
                            n_correct += 1
                            mw23_slot_value = normalize(list(co_ref_dict.keys())[0])
                            if not predicted_value == mw23_slot_value:
                                logging.warning(f"dataset mismatch: {dial_id}-{turn_id}, {slot_name}, "
                                                f"{predicted_value}, {mw23_slot_value}")
        accs.append(n_correct / n_total)
        total_coref_slots += n_total
    return np.mean(accs).item(), np.std(accs).item(), (total_coref_slots/len(runs))


def filter_each_run(runs: List[List[Turn]], filter: Callable[[List[Turn]], List[Turn]]) -> List[List[Turn]]:
    filtered = []
    for run in runs:
        filtered.append(filter(run))
    return filtered


def only_dialogues_with_coreference(turns: List[Turn], coreferences) -> List[Turn]:
    coref_dial_ids: Set[str] = set()
    for turn in turns:
        dial_id, turn_id = turn['ID'], turn['turn_id']
        # a dialogue is co-refferent if annotated as such AND the coref slots haven't been removed
        # in subsequent annotation cleanup (e.g. MultiWOZ 2.4)
        if dial_id in coreferences and any(
            slot_name in turn['slot_values'] for slot_name in coreferences[dial_id][turn_id]
        ):
            coref_dial_ids.add(dial_id)
    return [turn for turn in turns if turn['ID'] in coref_dial_ids]


def only_turns_with_coreference(turns: List[Turn], coreferences) -> List[Turn]:
    coref_turns: List[Turn] = []
    for turn in turns:
        dial_id, turn_id = turn['ID'], turn['turn_id']
        # a dialogue is co-refferent if annotated as such AND the coref slots haven't been removed
        # in subsequent annotation cleanup (e.g. MultiWOZ 2.4)
        if dial_id in coreferences and turn_id in coreferences[dial_id] and any(
            slot_name in turn['slot_values'] for slot_name in coreferences[dial_id][turn_id]
        ):
            coref_turns.append(turn)
    return coref_turns


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise ValueError("Need to specify a group")
    coreferences = get_coreference_annotations()
    group: str = sys.argv[1]
    print(f"Evaluating coreference for {group}")
    if "," in group:
        run_ids = group.split(",")
        runs = [read_run_artifact_logs(run_id) for run_id in run_ids]
    else:
        runs: List[List[Turn]] = get_running_logs_by_group(group_id=group)
    if not ("zero" in group or "_0p" in group):
        assert len(runs) == 3, "unexpected number of runs"
    else:
        assert len(runs) == 1, "unexpected number of runs"
    full_mean, full_std = eval_on_given_turns(runs)
    print(f"Full run performance => JGA:{full_mean:.2%}, (std={full_std:.2%})")
    coref_slots_mean, coref_slots_std, n_total = eval_just_coreference_slots(runs, coreferences)
    print(f"Coreference slot performance => Acc:{coref_slots_mean:.2%}, (std={coref_slots_std:.2%})")
    wandb_entity: str = os.environ.get(WANDB_ENTITY, "kingb12")
    wandb_project: str = os.environ.get(WANDB_PROJECT, "refpydst")
    run = wandb.init(project=wandb_project, entity=wandb_entity,
                     name=f"coreference_result_{group}", notes="eval_coref_turns.py", group="coreference_results")
    wandb.log({
        "full_mean": full_mean,
        "full_std": full_std,
        "coref_slots_mean": coref_slots_mean,
        "coref_slots_std": coref_slots_std,
        "total_coref_slots": n_total
    })

