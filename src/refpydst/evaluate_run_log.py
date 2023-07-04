import argparse
import json
from collections import defaultdict
from typing import Dict, List, Any

from refpydst.data_types import CompletionParser, Turn, MultiWOZDict
from tqdm import tqdm

from refpydst.error_analysis import read_running_log_file
from refpydst.evaluate_metrics import evaluate, evaluate_on_domain
from refpydst.normalization.abstract_normalizer import AbstractNormalizer
from refpydst.prompting import PROMPT_VARIANTS, IC_DST, get_completion_parser
from refpydst.utils.dialogue_state import update_dialogue_state
from refpydst.utils.state_recorder import PreviousStateRecorder


def evaluate_logs(running_log, test_set, turn=-1) -> Dict[str, Any]:
    # turn and use_gold are for analysis purpose
    # turn = -1 means evalute all dialogues
    # turn = 0 means evaluate single-turn dialogues
    # turn = 1 means evalute two-turn dialogues... etc.
    # when use_gold = True, the context are gold context (for analysis purpose)

    result_dict: Dict[int, List[int]] = defaultdict(list)  # use to record the accuracy

    # start experiment
    n_total = 0
    n_correct = 0
    total_acc = 0
    total_f1 = 0

    for data_item, label_item in tqdm(zip(running_log, test_set)):
        assert data_item['ID'] == label_item['ID'], \
            f"mismatched dialogues: {data_item['ID']}, {label_item['ID']}"
        assert data_item['turn_id'] == label_item['turn_id'], \
            f"mismatched dialogue turns: {data_item['turn_id']}, {label_item['turn_id']}"

        if turn >= 0:
            if data_item['turn_id'] != turn:
                continue

        n_total += 1

        # aggregate the prediction and the history states
        predicted_slot_values = data_item['pred']

        this_jga, this_acc, this_f1 = evaluate(
            predicted_slot_values, label_item['slot_values'])
        total_acc += this_acc
        total_f1 += this_f1

        if this_jga:
            n_correct += 1
            result_dict[data_item['turn_id']].append(1)
        else:
            result_dict[data_item['turn_id']].append(0)
    jga: float = n_correct / n_total
    slot_acc: float = total_acc / n_total
    joint_f1: float = total_f1 / n_total
    print(f"correct (JGA) {n_correct}/{n_total}  =  {jga:.4f}")
    print(f"Slot Acc {slot_acc:.4f}")
    print(f"Joint F1 {joint_f1:.4f}")
    print()

    # calculate the accuracy of each turn
    for k, v in result_dict.items():
        print(f"accuracy of turn {k} is {sum(v)}/{len(v)} = {sum(v) / len(v):.4f}")

    return {
        "slot_acc": slot_acc,
        "joint_f1": joint_f1,
        "jga": jga,
        "turn_accuracies": {k: sum(v) / len(v) for k, v in result_dict.items()}
    }


def replay_and_evaluate_logs(running_log, test_set, parser: CompletionParser, normalizer: AbstractNormalizer, turn=-1,
                             verify_pred: bool = False) -> Dict[str, Any]:
    # turn and use_gold are for analysis purpose
    # turn = -1 means evalute all dialogues
    # turn = 0 means evaluate single-turn dialogues
    # turn = 1 means evalute two-turn dialogues... etc.
    # when use_gold = True, the context are gold context (for analysis purpose)

    result_dict: Dict[int, List[int]] = defaultdict(list)  # use to record the accuracy

    # start experiment
    n_total = 0
    n_correct = 0
    total_acc = 0
    total_f1 = 0
    prediction_recorder: PreviousStateRecorder = PreviousStateRecorder()
    for data_item, label_item in tqdm(zip(running_log, test_set)):
        assert data_item['ID'] == label_item['ID'], \
            f"mismatched dialogues: {data_item['ID']}, {label_item['ID']}"
        assert data_item['turn_id'] == label_item['turn_id'], \
            f"mismatched dialogue turns: {data_item['turn_id']}, {label_item['turn_id']}"

        if turn >= 0:
            if data_item['turn_id'] != turn:
                continue

        n_total += 1

        # aggregate the prediction and the history states
        completion: str = data_item['completion']
        context: MultiWOZDict = prediction_recorder.retrieve_previous_turn_state(data_item)
        raw_parse: MultiWOZDict = parser(completion, context)
        normalized_parse: MultiWOZDict = normalizer.normalize(raw_parse)
        predicted_slot_values = update_dialogue_state(context, normalized_parse)
        prediction_recorder.add_state(data_item, predicted_slot_values)
        if verify_pred:
            assert data_item['pred'] == predicted_slot_values, "replayed prediction does not equal recorded prediction"

        this_jga, this_acc, this_f1 = evaluate(
            predicted_slot_values, label_item['slot_values'])
        total_acc += this_acc
        total_f1 += this_f1

        if this_jga:
            n_correct += 1
            result_dict[data_item['turn_id']].append(1)
        else:
            result_dict[data_item['turn_id']].append(0)
    jga: float = n_correct / n_total
    slot_acc: float = total_acc / n_total
    joint_f1: float = total_f1 / n_total
    print(f"correct (JGA) {n_correct}/{n_total}  =  {jga:.4f}")
    print(f"Slot Acc {slot_acc:.4f}")
    print(f"Joint F1 {joint_f1:.4f}")
    print()

    # calculate the accuracy of each turn
    for k, v in result_dict.items():
        print(f"accuracy of turn {k} is {sum(v)}/{len(v)} = {sum(v) / len(v):.4f}")

    return {
        "slot_acc": slot_acc,
        "joint_f1": joint_f1,
        "jga": jga,
        "turn_accuracies": {k: sum(v) / len(v) for k, v in result_dict.items()}
    }


def evaluate_on_domains(running_log: List[Turn], test_set: List[Turn], replay: bool = False,
                        replay_parser: CompletionParser = None, replay_normalizer: AbstractNormalizer = None) -> Dict[str, Dict[str, Any]]:
    scores: Dict[str, Dict[str, Any]] = {}
    for domain in ('attraction', 'hotel', 'taxi', 'train', 'restaurant'):
        assert len(running_log) == len(test_set), "number of logs does not match number of labels"
        n_total: int = 0
        n_correct: int = 0
        total_acc, total_f1 = 0, 0
        prediction_recorder: PreviousStateRecorder = PreviousStateRecorder()
        for data_item, label_item in tqdm(zip(running_log, test_set)):
            assert data_item['ID'] == label_item['ID'], f"dialogue IDs do not match: {data_item['ID']}, " \
                                                        f"{label_item['ID']}"
            assert data_item['turn_id'] == label_item['turn_id'], f"dialogue turn_ids do not match: " \
                                                                  f"{data_item['turn_id']}, {label_item['turn_id']}"
            if domain not in label_item['domains']:
                continue
            n_total += 1

            # aggregate the prediction and the history states
            if not replay:
                predicted_slot_values = data_item['pred']
            else:
                assert replay_parser and replay_normalizer, f"need to specify both a parser and normalizer. " \
                                                            f"Called with: {replay_parser}, {replay_normalizer}"
                completion: str = data_item['completion']
                context: MultiWOZDict = prediction_recorder.retrieve_previous_turn_state(data_item)
                raw_parse: MultiWOZDict = replay_parser(completion, context)
                normalized_parse: MultiWOZDict = replay_normalizer.normalize(raw_parse)
                predicted_slot_values = update_dialogue_state(context, normalized_parse)
                prediction_recorder.add_state(data_item, predicted_slot_values)


            # record current turn prediction
            this_jga, this_acc, this_f1 = evaluate_on_domain(predicted_slot_values, label_item['slot_values'], domain)
            total_acc += this_acc
            total_f1 += this_f1

            if this_jga:
                n_correct += 1
        scores[domain] = {'jga': n_correct / n_total, 'acc': total_acc / n_total, 'f1': total_f1 / n_total}
    return scores


if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--running_log', type=str, required=True,
                        help="running log filename")
    parser.add_argument('--prompt-format', type=str, default=IC_DST, choices=PROMPT_VARIANTS,
                        help="prompt format (needed to determine how to parse the completion)")
    parser.add_argument('--test_fn', type=str, default="./data/mw24_100p_test.json",
                        help="running log filename")
    parser.add_argument('--mwz_ver', type=str, default="2.4",
                        choices=['2.1', '2.4'], help="version of MultiWOZ")
    parser.add_argument('--use_gold', type=bool, default=False,
                        help="whether to use gold contexts for evaluating deltas. False shows true system behavior, " \
                             "True shows which turns include incorrect deltas")
    args = parser.parse_args()

    # read the running log
    running_log = read_running_log_file(args.running_log)

    # read the testing file
    with open(args.test_fn) as f:
        test_set = json.load(f)

    completion_parser: CompletionParser
    completion_parser = get_completion_parser(args.prompt_format)
    evaluate_logs(running_log, test_set)
