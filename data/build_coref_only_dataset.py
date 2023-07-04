import json
from typing import Any, Dict

from refpydst.data.multiwoz23 import get_coreference_annotations
from refpydst.utils.dialogue_state import group_by_dial_id_and_turn
from refpydst.utils.general import read_json

if __name__ == '__main__':
    dev_set = read_json("mw24_100p_dev.json")

    coreferences: Dict[str, Any] = get_coreference_annotations()
    only_coref_turns = [dial for dial in dev_set if dial['ID'] in coreferences]
    dev_by_dial_ids = group_by_dial_id_and_turn(dev_set)
    coref_only_by_dial_ids = group_by_dial_id_and_turn(only_coref_turns)
    for k in coref_only_by_dial_ids:
        assert len(coref_only_by_dial_ids[k]) == len(dev_by_dial_ids[k])
    with open("mw24_coref_only_dials_dev.json", "w") as f:
        json.dump(only_coref_turns, f)
