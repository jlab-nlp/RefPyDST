import copy
from collections import defaultdict
from typing import Dict, List

import dictdiffer
from refpydst.data_types import MultiWOZDict, Turn


def compute_delta(prev_dst: MultiWOZDict, dst: MultiWOZDict) -> MultiWOZDict:
    """
    Compute the difference between two dialogue states, each in flattened form
    
    :param prev_dst: previous dialogue state (e.g. prior turn) 
    :param dst: current dialogue state (e.g. this turn)
    :return: difference, as its own dialogue state
    """
    delta: MultiWOZDict = {}
    for diff in dictdiffer.diff(prev_dst, dst):
        # initial step: treat changes as adds, since we're starting from an empty dict (UPSERT like behavior)
        if diff[0] == 'change':
            diff = ('add', '', [(diff[1], diff[2][1])])
        # apply adds
        if diff[0] == 'add':
            delta = dictdiffer.patch([diff], delta)
        elif diff[0] == 'remove':
            assert diff[1] == ''  # code won't work otherwise, but should be true of our representation
            for slot_name, slot_value in diff[2]:
                delta[slot_name] = '[DELETE]'
    return delta


def equal_dialogue_utterances(turn_a: Turn, turn_b: Turn) -> bool:
    for key in ('sys', 'usr'):
        if len(turn_a['dialog'][key]) != len(turn_b['dialog'][key]):
            return False
        for utterance_a, utterance_b in zip(turn_a['dialog'][key], turn_b['dialog'][key]):
            if utterance_a != utterance_b:
                return False
    return True


def update_dialogue_state(context: MultiWOZDict, normalized_turn_parse: MultiWOZDict) -> MultiWOZDict:
    """
    Given a normalized parse for a turn state and an existing prior state, compute the new complete
    updated state.

    :param context: complete state at turn t - 1
    :param normalized_turn_parse: predicted state change at turn t
    :return: complete state at turn t
    """
    new_dialogue_state: MultiWOZDict = copy.deepcopy(context)
    for slot_name, slot_value in normalized_turn_parse.items():
        if slot_name in new_dialogue_state and slot_value == "[DELETE]":
            del new_dialogue_state[slot_name]
        elif slot_value != "[DELETE]":
            new_dialogue_state[slot_name] = slot_value
    return new_dialogue_state


def group_by_dial_id_and_turn(turns: List[Turn]) -> Dict[str, List[Turn]]:
    result = defaultdict(dict)
    for turn in turns:
        result[turn['ID']][turn['turn_id']] = turn
    return {dial_id: [turn for index, turn in sorted(turns_dict.items(), key=lambda item: item[0])]
            for dial_id, turns_dict in result.items()}
