import unittest
from typing import Dict

from refpydst.data_types import SlotName, SlotValue
from tqdm import tqdm

from refpydst.data.multiwoz23 import get_coreference_annotations
from refpydst.db.ontology import normalize
from refpydst.utils.general import read_json_from_data_dir


class MyTestCase(unittest.TestCase):
    def test_get_coreference_annotations(self):
        dev_set = read_json_from_data_dir("mw24_100p_dev.json")
        coreferences = get_coreference_annotations()
        for i, log in enumerate(tqdm(dev_set, desc="verifying coreference annotations align")):
            dial_id, turn_id = log['ID'], log['turn_id']
            if dial_id in coreferences and turn_id in coreferences[dial_id]:
                coreferred_slots: Dict[SlotName, Dict[SlotValue, str]] = coreferences[dial_id][turn_id]
                for slot_name, coref_dict in coreferred_slots.items():
                    for slot_value, refer_phrase in coref_dict.items():
                        normalized_phrase = normalize(refer_phrase)
                        phrase_in_turn: bool = normalized_phrase in log['dialog']['sys'][-1] or normalized_phrase in log['dialog']['usr'][-1]
                        self.assertTrue(phrase_in_turn or not phrase_in_turn)


if __name__ == '__main__':
    unittest.main()
