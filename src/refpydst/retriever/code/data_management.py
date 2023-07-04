from typing import Callable, Union, List

import numpy as np
from refpydst.data_types import Turn, MultiWOZDict
from tqdm import tqdm

from refpydst.prompt_formats.python.demo import get_state_reference
from refpydst.prompt_formats.python.demo import normalize_to_domains_and_slots, SLOT_NAME_REVERSE_REPLACEMENTS
from refpydst.retriever.code.retriever_evaluation import compute_sv_sim
from refpydst.utils.general import read_json_from_data_dir

# Only care domain in test

DOMAINS = ['hotel', 'restaurant', 'attraction', 'taxi', 'train']


def important_value_to_string(slot, value):
    if value in ["none", "dontcare"]:
        return f"{slot}{value}"  # special slot
    return f"{slot}-{value}"


StateTransformationFunction = Callable[[Turn], Union[List[str], MultiWOZDict]]


def default_state_transform(turn: Turn) -> List[str]:
    return [important_value_to_string(s, v) for s, v in turn['turn_slot_values'].items()
            if s.split('-')[0] in DOMAINS]


def reference_aware_state_transform(turn: Turn) -> List[str]:
    context_slot_values: MultiWOZDict = {s: v.split('|')[0] for s, v in turn['last_slot_values'].items()}
    context_domains_to_slot_pairs = normalize_to_domains_and_slots(context_slot_values)
    turn_domains_to_slot_pairs = normalize_to_domains_and_slots(turn['turn_slot_values'])
    user_str = turn['dialog']['usr'][-1]
    sys_str = turn['dialog']['sys'][-1]
    sys_str = sys_str if sys_str and not sys_str == 'none' else ''
    new_dict: MultiWOZDict = {}
    for domain, slot_pairs in turn_domains_to_slot_pairs.items():
        for norm_slot_name, norm_slot_value in slot_pairs.items():
            # Because these come from a prompting template set up, the slot names were changed in a few cases,
            # and integer-string values were turned into actual integers
            mwoz_slot_name = SLOT_NAME_REVERSE_REPLACEMENTS.get(norm_slot_name, norm_slot_name.replace("_", " "))
            mwoz_slot_value = str(norm_slot_value)
            reference = get_state_reference(context_domains_to_slot_pairs, domain, norm_slot_name, norm_slot_value,
                                            turn_strings=[sys_str, user_str])
            if reference is not None:
                referred_domain, referred_slot = reference
                new_dict[f"{domain}-{mwoz_slot_name}"] = f"state.{referred_domain}.{referred_slot}"
            else:
                new_dict[f"{domain}-{mwoz_slot_name}"] = mwoz_slot_value
    return [important_value_to_string(s, v) for s, v in new_dict.items()
            if s.split('-')[0] in DOMAINS]


def get_state_transformation_by_type(tranformation_type: str) -> StateTransformationFunction:
    if not tranformation_type or tranformation_type == "default":
        return default_state_transform
    elif tranformation_type == "ref_aware":
        return reference_aware_state_transform
    else:
        raise ValueError(f"Unsupported transformation type: {tranformation_type}")


def get_string_transformation_by_type(tranformation_type: str):
    if not tranformation_type or tranformation_type == "default":
        return data_item_to_string
    else:
        raise ValueError(f"Unsupported transformation type: {tranformation_type}")


def input_to_string(context_dict, sys_utt, usr_utt):
    history = state_to_NL(context_dict)
    if sys_utt == 'none':
        sys_utt = ''
    if usr_utt == 'none':
        usr_utt = ''
    history += f" [SYS] {sys_utt} [USER] {usr_utt}"
    return history


def data_item_to_string(data_item: Turn, string_transformation=input_to_string, full_history: bool = False) -> str:
    """
    Converts a turn to a string with the context, system utterance, and user utterance in order

    :param data_item: turn to use
    :param string_transformation: function defining how to represent the context, system utterance, user utterance
           triplet as a single string
    :param full_history: use the complete dialogue history, not just current turn
    :return: string representation, like below


    Example (new lines and tabs added for readability):
    [CONTEXT] attraction name: saint johns college,
    [SYS] saint john s college is in the centre of town on saint john s street . the entrance fee is 2.50 pounds .
          can i help you with anything else ?
    [USER] is there an exact address , like a street number ? thanks !
    """

    # use full history, depend on retriever training (for ablation)
    if full_history:
        history = ""
        for sys_utt, usr_utt in zip(data_item['dialog']['sys'], data_item['dialog']['usr']):
            history += string_transformation({}, sys_utt, usr_utt)
        return history

    # use single turn
    context = data_item['last_slot_values']
    sys_utt = data_item['dialog']['sys'][-1]
    usr_utt = data_item['dialog']['usr'][-1]
    history = string_transformation(context, sys_utt, usr_utt)
    return history


def state_to_NL(slot_value_dict):
    output = "[CONTEXT] "
    for k, v in slot_value_dict.items():
        output += f"{' '.join(k.split('-'))}: {v.split('|')[0]}, "
    return output


class MWDataset:

    def __init__(self, mw_json_fn: str, just_embed_all=False, beta: float = 1.0,
                 string_transformation: Callable[[Turn], str] = data_item_to_string,
                 state_transformation: StateTransformationFunction = default_state_transform):

        data = read_json_from_data_dir(mw_json_fn)

        self.turn_labels = []  # store [SMUL1843.json_turn_1, ]
        self.turn_utts = []  # store corresponding text
        self.turn_states = []  # store corresponding states. [['attraction-type-mueseum',],]
        self.string_transformation = string_transformation
        self.state_transformation = state_transformation

        for turn in data:
            # filter the domains that not belongs to the test domain
            if not set(turn["domains"]).issubset(set(DOMAINS)):
                continue

            # update dialogue history
            history = self.string_transformation(turn)

            # convert to list of strings
            current_state = self.state_transformation(turn)

            self.turn_labels.append(f"{turn['ID']}_turn_{turn['turn_id']}")
            self.turn_utts.append(history)
            self.turn_states.append(current_state)

        self.n_turns = len(self.turn_labels)
        print(f"there are {self.n_turns} turns in this dataset")

        if not just_embed_all:
            # compute all similarity
            self.similarity_matrix = np.zeros((self.n_turns, self.n_turns))
            for i in tqdm(range(self.n_turns)):
                self.similarity_matrix[i, i] = 1
                for j in range(i, self.n_turns):
                    self.similarity_matrix[i, j] = compute_sv_sim(self.turn_states[i],
                                                                  self.turn_states[j],
                                                                  beta=beta)
                    self.similarity_matrix[j, i] = self.similarity_matrix[i, j]


def save_embeddings(model, dataset: MWDataset, output_filename: str) -> None:
    """
    Save embeddings for all items in the dataset to the output_filename

    :param dataset: dataset to create and save embeddings from
    :param output_filename: path to the save file
    :return: None
    """
    embeddings = model.encode(dataset.turn_utts, convert_to_numpy=True)
    output = {}
    for i in tqdm(range(len(embeddings))):
        output[dataset.turn_labels[i]] = embeddings[i:i + 1]
    np.save(output_filename, output)
