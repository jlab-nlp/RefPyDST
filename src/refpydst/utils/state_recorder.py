"""
This file was adapted from the code for the paper "In Context Learning for Dialogue State Tracking", as originally
published here: https://github.com/Yushi-Hu/IC-DST. Cite their article as:

@article{hu2022context,
  title={In-Context Learning for Few-Shot Dialogue State Tracking},
  author={Hu, Yushi and Lee, Chia-Hsuan and Xie, Tianbao and Yu, Tao and Smith, Noah A and Ostendorf, Mari},
  journal={arXiv preprint arXiv:2203.08568},
  year={2022}
}
"""
from collections import defaultdict
from typing import Dict

from refpydst.data_types import MultiWOZDict, Turn


class PreviousStateRecorder:
    """
    Records predictions for use in subsequent turns
    """
    # we'll just track these in memory for now
    states: Dict[str, Dict[int, MultiWOZDict]]

    def __init__(self):
        self.states = defaultdict(dict)

    def add_state(self, turn: Turn, slot_values: MultiWOZDict) -> None:
        """
        Store the slot values for a given turn, indexable by dialogue and turn_id
        :param turn: 
        :param slot_values: 
        :return: 
        """
        dialog_id: str = turn['ID']
        turn_id: int = turn['turn_id']
        self.states[dialog_id][turn_id] = slot_values

    def retrieve_previous_turn_state(self, turn: Turn) -> MultiWOZDict:
        """
        retrieve a recorded prediction for the turn prior to this one. Raises an exception if missing, implying we did 
        not process turns in order.
        
        :param turn: 
        :return: 
        """
        dialog_id = turn['ID']
        turn_id = turn['turn_id']
        if turn_id == 0:
            return {}
        else:
            return self.states[dialog_id][turn_id - 1]