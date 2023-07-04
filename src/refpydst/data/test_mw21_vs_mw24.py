import unittest

from tqdm import tqdm

from refpydst.utils.dialogue_state import equal_dialogue_utterances
from refpydst.utils.general import read_json_from_data_dir

"""
Simple verifications that we can use predictions from a run on MultiWOZ 2.4 to evaluate on MultiWOZ 2.1, without
repeating the experiment: inputs need to match. Given these, we can evaluate by just comparing our prediction to the 
labels in both versions.
"""


class MyTestCase(unittest.TestCase):
    def test_equivalent_dev_set_inputs(self):
        dev_set_21 = read_json_from_data_dir("mw21_100p_dev.json")
        dev_set_24 = read_json_from_data_dir("mw24_100p_dev.json")
        self.assertEqual(len(dev_set_21), len(dev_set_24))
        for turn_21, turn_24 in tqdm(zip(dev_set_21, dev_set_24), desc="validating equivalent dialogue inputs"):
            # turn identifiers should be equal
            self.assertEqual(turn_21['ID'], turn_24['ID'])
            self.assertEqual(turn_21['turn_id'], turn_24['turn_id'])
            # all dialog utterances should be equal
            self.assertTrue(equal_dialogue_utterances(turn_21, turn_24))
            for key in ('sys', 'usr'):
                self.assertEqual(len(turn_21['dialog'][key]), len(turn_24['dialog'][key]))
                for utterance_21, utterance_24 in zip(turn_21['dialog'][key], turn_24['dialog'][key]):
                    self.assertEqual(utterance_21, utterance_24)

    # necessary for verifying that we can evaluate on 2.1 and 2.4 with a single run
    def test_equivalent_test_set_inputs(self):
        test_set_21 = read_json_from_data_dir("mw21_100p_test.json")
        test_set_24 = read_json_from_data_dir("mw24_100p_test.json")
        self.assertEqual(len(test_set_21), len(test_set_24))
        for turn_21, turn_24 in tqdm(zip(test_set_21, test_set_24), desc="validating equivalent dialogue inputs"):
            # turn identifiers should be equal
            self.assertEqual(turn_21['ID'], turn_24['ID'])
            self.assertEqual(turn_21['turn_id'], turn_24['turn_id'])
            # all dialog utterances should be equal
            self.assertTrue(equal_dialogue_utterances(turn_21, turn_24))
            for key in ('sys', 'usr'):
                self.assertEqual(len(turn_21['dialog'][key]), len(turn_24['dialog'][key]))
                for utterance_21, utterance_24 in zip(turn_21['dialog'][key], turn_24['dialog'][key]):
                    self.assertEqual(utterance_21, utterance_24)


if __name__ == '__main__':
    unittest.main()
