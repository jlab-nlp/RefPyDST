import unittest
from typing import List

from refpydst.data_types import Turn
from tqdm import tqdm

from refpydst.prompt_formats.python.demo import normalize_to_domains_and_slots, DOMAIN_CLASSES, \
    _remove_quotes_on_references
from refpydst.utils.general import read_json_from_data_dir


class TestDemoCreator(unittest.TestCase):
    def test_slot_alignment(self):
        data: List[Turn] = read_json_from_data_dir("mw24_10p_dev.json")
        for turn in tqdm(data, desc="validating turn slot-values against prompt format", total=len(data)):
            normalized = normalize_to_domains_and_slots(turn['slot_values'])
            for domain in normalized:
                assert domain in DOMAIN_CLASSES, domain
                clazz = DOMAIN_CLASSES[domain]
                for slot_name, slot_value in normalized[domain].items():
                    assert hasattr(clazz, slot_name), f"{clazz}, {slot_name}"

    def test_re_replace_state_string(self):
        self.assertEqual(
            'BeliefState.from_dict({"hotel": {"name": state.hotel.name}}',
            _remove_quotes_on_references('BeliefState.from_dict({"hotel": {"name": "state.hotel.name"}}'),
        )
        self.assertEqual(
            'BeliefState.from_dict({"hotel": {"name": state.hotel.name, "area": "centre"}}',
            _remove_quotes_on_references('BeliefState.from_dict({"hotel": {"name": "state.hotel.name", "area": "centre"}}'),
        )
        self.assertEqual(
            'BeliefState.from_dict({"hotel": {"name": state.hotel.name, "name_2": state.hotel.name}}',
            _remove_quotes_on_references(
                'BeliefState.from_dict({"hotel": {"name": "state.hotel.name", "name_2": "state.hotel.name"}}'),
        )

if __name__ == '__main__':
    unittest.main()
