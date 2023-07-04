import copy
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, get_type_hints, Optional, Final, Any

from refpydst.data_types import SlotValue, MultiWOZDict, SlotValuesByDomain

from refpydst.db.ontology import CATEGORICAL_SLOT_VALUE_TYPES
from refpydst.prompt_formats.python.python_classes import Hotel, Train, Attraction, Restaurant, Taxi, Option
from refpydst.utils.dialogue_state import compute_delta


# these classes aren't evaluated in MultiWOZ, and so they don't come from the prompt
@dataclass
class Police:
    name: str = None


@dataclass
class Hospital:
    department: str = None


DOMAIN_CLASSES: Dict[str, Any] = {
    "hotel": Hotel,
    "train": Train,
    "attraction": Attraction,
    "restaurant": Restaurant,
    "taxi": Taxi,
    "hospital": Hospital,
    "police": Police
}
DOMAIN_IDX: Dict[str, int] = {
    "hotel": 0,
    "train": 1,
    "attraction": 2,
    "restaurant": 3,
    "taxi": 4,
    "hospital": 5,
    "police": 6
}

SLOT_NAME_REPLACEMENTS: Dict[str, str] = {
    "leaveat": "depart_time",
    "arriveby": "arrive_by_time",
    "book stay": "book_number_of_days",
    "food": "food_type",
    "departure": "leave_from",
    "pricerange": "price_range"
}
SLOT_NAME_REVERSE_REPLACEMENTS: Dict[str, str] = {v: k for k, v in SLOT_NAME_REPLACEMENTS.items()}


def get_user_string(user_utterance: str) -> str:
    return f'print("user: {user_utterance}")'


def get_system_string(sys_utterance: str) -> str:
    return f'print("system: {sys_utterance}")'


def fix_general_typos(value: str) -> str:
    # fix typo words
    general_typos = {'fen ditton': 'fenditton',
                     'guesthouse': 'guest house',
                     'steveage': 'stevenage',
                     'stantsted': 'stansted',
                     'storthford': 'stortford',
                     'shortford': 'stortford',
                     'weish': 'welsh',
                     'bringham': 'birmingham',
                     'birminggam': 'birmingham',
                     'boxbourne': 'broxbourne',
                     'gardina': 'gardiena',
                     'liverpoool': 'liverpool',
                     'petersborough': 'peterborough',
                     'el shaddai': 'el shaddia',
                     'wendesday': 'wednesday',
                     'brazliian': 'brazilian',
                     'graffton': 'grafton'}
    for k, v in general_typos.items():
        value = value.replace(k, v)
    return value


def _quote_if_needed(domain: str, slot_name: str, slot_value: SlotValue) -> str:
    slot_type_hint = get_type_hints(DOMAIN_CLASSES[domain])[slot_name]
    if type(slot_value) == str and (slot_value.startswith("agent.state") or slot_value.startswith("state.")):
        return str(slot_value)  # this is a reference, don't quote wrap
    elif slot_type_hint != int:
        return f"\"{slot_value}\""  # a string or Literal of some kind, etc.
    else:
        return str(slot_value)  # an int or numeric: don't quote wrap


def get_state_reference(prior_state_norm: Dict[str, Dict[str, SlotValue]], domain: str, slot_name: str,
                        slot_value: SlotValue, turn_strings: List[str]) -> Optional[Tuple[str, str]]:
    # dontcare is not a co-referable dialogue item in practice (a user would not really say I feel about the price-range
    # the same way I do about the hotel-type as an indication of "dontcare")
    if slot_value == "dontcare":
        return None

    # helper: return true if this slot-pair refers to a number or time
    def _is_numeric_or_bool(domain: str, slot_name: str) -> bool:
        slot_type_hint = get_type_hints(DOMAIN_CLASSES[domain])[slot_name]
        return slot_type_hint == int or slot_type_hint == Option

    if _is_numeric_or_bool(domain, slot_name):
        # this is a number or a boolean. For this data-type, meaning is dependent on the slot name (i.e. I would refer
        # to the same number of people, not "5" in abstract. I could, but would be unlikely to say "book me hotel
        # for the same number of nights as people in my restaurant reservation")
        #
        # Luckily, for all integer and time based slots, MultiWOZ uses the same slot_name for the same kind of meaning.
        # This makes programming this rule easier, but it should still be generally doable in other ontologies, since we
        # are making this determination based two gold state dictionaries, not an un-labelled turn.
        for state_domain, state_pairs in prior_state_norm.items():
            # for a slot-pair that is numeric/bool to be considered as co-referring to something in the state:
            # 1) the slot value must exactly match the slot value for some other slot in the state
            # 2) the number/bool must NOT be in either of the turn strings
            # 3) the slot-name (excluding domain) must match
            # i.e. we choose to favor coincidental numeral equivalence as NOT co-reference, i.e. (hotel-stay, 6) is
            # unlikely to co-refer to (restaurant-book people, 6), even if we can't find an occurrence of 6 in the turn
            # strings: 6 nights != 6 people
            if slot_name in state_pairs and not any(f" {slot_value} " in s for s in turn_strings) and \
                    slot_value == state_pairs[slot_name]:
                return state_domain, slot_name
        # it was a numeric, but there was no occurrence in the state
        return None

    # next check for explicit mentions of the slot value. For numerics, these can be false positive if there is overlap
    # on value, hence the prior check. For names, days of the week, etc, this is less likely
    if any(str(slot_value) in fix_general_typos(s) for s in turn_strings):
        # this slot-value is explicitly mentioned in the turn. This will misinterpret statements like the following:
        # state = {restaurant-book_people = 2} "i need a hotel for the same group of people for 2 nights "
        return None

    # finally, check if the slot is mentioned in the state. Since we're comparing two normalized states, ideally we
    # shouldn't need to do any fuzzy string matching (still, we'll ignore casing)
    for state_domain, state_pairs in prior_state_norm.items():
        for state_slot_name, state_value in state_pairs.items():
            if type(state_value) == str and type(slot_value) == str and slot_value.lower() == state_value.lower():
                # double check a couple of specifics:
                if slot_name == "area" and slot_value == "centre" and any("center" in s for s in turn_strings):
                    return None
                return state_domain, state_slot_name
    return None


def get_python_statements(prior_state: MultiWOZDict, gold_state: MultiWOZDict,
                          turn_strings: List[str], detailed_state_string: bool = False) -> Tuple[str, str]:
    # first calculate and write a state string for the prior state
    prior_state_norm: SlotValuesByDomain = normalize_to_domains_and_slots(prior_state)
    if not detailed_state_string:
        domains_str: str = f"domains=[{', '.join(DOMAIN_CLASSES[domain].__name__ for domain in prior_state_norm)}]"
        state_str: Final[str] = f"agent.state = BeliefState.from_dialogue_history({domains_str})"
    else:
        state_str: Final[str] = get_state_string(prior_state, delexicalize_non_categoricals=True)
    delta = compute_delta(prior_state, gold_state)
    delta = normalize_to_domains_and_slots(delta)
    deletions: List[Tuple[str, str]] = []
    upserts: SlotValuesByDomain = defaultdict(dict)
    for domain, slot_pairs in delta.items():
        for slot_name, slot_value in slot_pairs.items():
            if slot_value == "[DELETE]":
                deletions.append((domain, slot_name))
            else:
                upserts[domain][slot_name] = slot_value
    state_change_lines: List[str] = []
    for domain, slot_name in deletions:
        state_change_lines.append(f"del agent.state.{domain}.{slot_name}")
    for domain, slot_pairs in upserts.items():
        for slot_name, slot_value in slot_pairs.items():
            reference = get_state_reference(prior_state_norm, domain, slot_name, slot_value,
                                            turn_strings=turn_strings)
            if reference is not None:
                # over-write the slot value with a reference to the slot in the current state
                referred_domain, referred_slot = reference
                slot_pairs[slot_name] = f"agent.state.{referred_domain}.{referred_slot}"
        state_change_lines.append(
            f"agent.state.{domain} = agent.find_{domain}" +
            f"({', '.join(name + '=' + _quote_if_needed(domain, name, value) for name, value in slot_pairs.items())})"
        )
    if not state_change_lines:
        state_change_lines.append("agent.state.update({})  # no change")
    return state_str, "\n".join(state_change_lines)


def get_state_string(prior_state: MultiWOZDict, delexicalize_non_categoricals: bool = False) -> str:
    """
    Return a string representing a line for the prior state, as could be included in a prompt

    :param prior_state: dialogue prior state to represent
    :param delexicalize_non_categoricals: if True, replace each non-categorical value with a pseudo "state-reference"
    :return: string representing the state, on a single line
    """
    state_dict: MultiWOZDict = copy.deepcopy(prior_state)
    if delexicalize_non_categoricals:
        for slot_name, slot_value in prior_state.items():
            domain, slot_type = slot_name.split("-")  # hotel-area => hotel, area
            if slot_type not in CATEGORICAL_SLOT_VALUE_TYPES:
                # delexicalize to a state reference
                state_dict[slot_name] = f"state.{domain}.{SLOT_NAME_REPLACEMENTS.get(slot_type, slot_type.replace(' ', '_'))}"
    prior_state_norm: SlotValuesByDomain = normalize_to_domains_and_slots(state_dict)
    state_str: str = f"agent.state = BeliefState.from_dict({json.dumps(prior_state_norm)})"
    return _remove_quotes_on_references(state_str)


def _remove_quotes_on_references(state_str: str) -> str:
    return re.sub(r"\"(state\..*?)\"", repl=r"\1", string=state_str)


def normalize_to_domains_and_slots(state: Dict[str, SlotValue]) -> SlotValuesByDomain:
    normalized = defaultdict(dict)
    for domain_and_slot_name, slot_value in state.items():
        domain, slot_name = domain_and_slot_name.split("-")
        slot_name = SLOT_NAME_REPLACEMENTS.get(slot_name, slot_name.replace(" ", "_"))
        normalized[domain][slot_name] = _make_int_slot_value(slot_value)
    return dict(sorted(normalized.items(), key=lambda domain_and_slots: DOMAIN_IDX.get(domain_and_slots[0], 100)))


def _make_int_slot_value(s: str) -> SlotValue:
    try:
        return int(s)
    except ValueError:
        return s
