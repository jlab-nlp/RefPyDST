import enum
import json
import logging
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Callable

import numpy as np
import scipy.stats as st
from numpy.typing import NDArray
from refpydst.data_types import Turn
from scipy.stats.morestats import WilcoxonResult
from tqdm import tqdm

from refpydst.artifacts import get_json_artifact_by_file_name
from refpydst.evaluate_metrics import evaluate


# inheriting from str + enum.Enum allows for painless JSON serialization
class F1Class(str, enum.Enum):
    TP = 'tp'
    FP = 'fp'
    FN = 'fn'


def read_running_log_file(file_name: str) -> List[Turn]:
    try:
        return get_json_artifact_by_file_name(file_name)
    except BaseException as e:
        logging.warning(e)
        with open(file_name, 'r') as f:
            return json.load(f)


def evaluate_logs(logs: List[Turn], gold_key: str = 'slot_values') -> List[Turn]:
    for turn in tqdm(logs, desc="evaluating turns", total=len(logs)):
        this_jga, this_acc, this_f1 = evaluate(turn['pred'], turn[gold_key])
        turn['jga'] = this_jga
        turn['acc'] = this_acc
        turn['turn_f1'] = this_f1
    return logs


def calc_f1(counter: Counter[F1Class]) -> float:
    precision: float = calc_precision(counter)
    recall: float = calc_recall(counter)
    if precision and recall:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0


def calc_recall(counter: Counter[F1Class]) -> float:
    recall_denom: int = (counter[F1Class.TP] + counter[F1Class.FN])
    recall: float = counter[F1Class.TP] / recall_denom if recall_denom else 0
    return recall


def calc_precision(counter: Counter[F1Class]) -> float:
    precision_denom: int = (counter[F1Class.TP] + counter[F1Class.FP])
    precision: float = counter[F1Class.TP] / precision_denom if precision_denom else 0
    return precision


def slot_level_f1(logs: List[Turn], tp_means_correct: bool = True, gold_key: str = 'slot_values') -> Dict[
    str, Tuple[Counter[F1Class], float]]:
    slot_scores: Dict[str, Counter[F1Class]] = defaultdict(Counter)
    for turn in tqdm(logs, desc="calculating slot-level F1", total=len(logs)):
        for gold_slot, gold_value in turn[gold_key].items():
            # if the slot is present in prediction, whether its a TP or FN depends on our tp_means_correct flag
            if gold_slot in turn['pred'] and (not tp_means_correct or turn['pred'][gold_slot] == gold_value):
                slot_scores[gold_slot][F1Class.TP] += 1
            else:
                slot_scores[gold_slot][F1Class.FN] += 1
        for pred_slot, pred_value in turn['pred'].items():
            if pred_slot not in turn[gold_key]:
                slot_scores[pred_slot][F1Class.FP] += 1
    return {k: (v, calc_f1(v)) for k, v in slot_scores.items()}


def count_prompts_from_examples(examples: List[Turn]) -> Counter[str]:
    """
    Given a list of example turns that were actually used in a prompt, return the number of demonstrated completions
    of each slot, if any.

    :param examples: list of demonstrations
    :return: dict from MultiWOZ normalized slot name to count
    """
    prompt_counter: Counter = Counter()
    for demo_turn in examples:
        for slot_name in demo_turn['turn_slot_values']:
            prompt_counter[slot_name] += 1
    return prompt_counter


def _mean_jga(turns: List[Turn]) -> float:
    return np.mean([t['jga'] for t in turns]).item()


def permutation_test(base_log_runs: List[List[Turn]], adj_log_runs: List[List[Turn]],
                     dialogue_metric: Callable[[List[Turn]], float] = _mean_jga) -> float:
    # Across N runs, group turns by dialogue id and run #
    base_dials: Dict[Tuple[int, str], List[Turn]] = defaultdict(list)
    adj_dials: Dict[Tuple[int, str], List[Turn]] = defaultdict(list)

    for run_idx, (base_logs, adj_logs) in enumerate(zip(base_log_runs, adj_log_runs)):
        # Group each by dialogue id
        for turn in base_logs:
            base_dials[(run_idx, turn['ID'])].append(turn)
        for turn in adj_logs:
            adj_dials[(run_idx, turn['ID'])].append(turn)

    x = np.array([dialogue_metric(turns) for k, turns in adj_dials.items()])
    y = np.array([dialogue_metric(turns) for k, turns in base_dials.items()])

    def _mean_diff(x: NDArray, y: NDArray) -> float:
        return np.mean(x) - np.mean(y)

    result = st.permutation_test([x, y], _mean_diff, permutation_type='samples', alternative="greater",n_resamples=99999)
    return result.pvalue


def wilcox_sign_ranked_test(base_log_runs: List[List[Turn]], adj_log_runs: List[List[Turn]],
                            dialogue_metric: Callable[[List[Turn]], float] = _mean_jga) -> float:
    """
    Given a system A and B evaluated on a PAIRED list of turns, return the probability that the
    difference in performance is supported by the null hypothesis. Specifically, we group turns by dialogue, as
    turns are not i.i.d or exchangeable but dialogues are.

    :param base_log_runs: sequence of turns from the baseline
    :param adj_log_runs: sequence of turns from the adjustment
    :return: the p-value indicating the probability that the baseline performance is greater than or equal to the
      performance of the adjustment
    """
    # Across N runs, group turns by dialogue id and run #
    base_dials: Dict[Tuple[int, str], List[Turn]] = defaultdict(list)
    adj_dials: Dict[Tuple[int, str], List[Turn]] = defaultdict(list)

    for run_idx, (base_logs, adj_logs) in enumerate(zip(base_log_runs, adj_log_runs)):
        # Group each by dialogue id
        for turn in base_logs:
            base_dials[(run_idx, turn['ID'])].append(turn)
        for turn in adj_logs:
            adj_dials[(run_idx, turn['ID'])].append(turn)

    x = [dialogue_metric(turns) for k, turns in adj_dials.items()]
    y = [dialogue_metric(turns) for k, turns in base_dials.items()]

    # The wilcoxon signed rank test generally
    result: WilcoxonResult = st.wilcoxon(x=x, y=y, zero_method="zsplit", alternative="greater")
    return result.pvalue
