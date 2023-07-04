import re
import unittest
from typing import List

from refpydst.data_types import Turn
from tqdm import tqdm

from refpydst.prompt_formats.python.demo import SLOT_NAME_REVERSE_REPLACEMENTS
from refpydst.retriever.code.data_management import default_state_transform, reference_aware_state_transform
from refpydst.utils.general import read_json_from_data_dir


class MyTestCase(unittest.TestCase):
    def test_state_transformations_are_same_when_expected(self):
        turns: List[Turn] = read_json_from_data_dir("mw24_100p_dev.json")
        for i, turn in tqdm(enumerate(turns)):
            default_transformed = default_state_transform(turn)
            ref_aware_transformed = reference_aware_state_transform(turn)
            # if no references, these should be the same (order doesn't matter)
            if not any("state." in slot_pair_str for slot_pair_str in ref_aware_transformed):
                self.assertCountEqual(default_transformed, ref_aware_transformed,
                                      msg="expected identical state representations")
            else:
                dereferenced = []
                for slot_pair_str in ref_aware_transformed:
                    if not "state." in slot_pair_str:
                        dereferenced.append(slot_pair_str)
                    else:
                        _, domain, slot_name = slot_pair_str.split(".")
                        un_normed_slot_name = SLOT_NAME_REVERSE_REPLACEMENTS.get(slot_name, slot_name.replace("_", " "))
                        referred_value = turn['slot_values'][f"{domain}-{un_normed_slot_name}"]
                        dereferenced.append(re.sub(r"state\..*$", referred_value, slot_pair_str))
                # we should be able to reconstruct with referred values (order doesn't matter)
                self.assertCountEqual(default_transformed, dereferenced)


if __name__ == '__main__':
    unittest.main()
