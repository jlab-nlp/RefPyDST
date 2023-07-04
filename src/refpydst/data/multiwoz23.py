# data.json from extracted files in MultiWOZ 2.3 dataset repo:
# https://github.com/lexmen318/MultiWOZ-coref/raw/main/MultiWOZ2_3.zip

from collections import defaultdict
from typing import Dict, Any

from refpydst.data_types import SlotName
from tqdm import tqdm

from refpydst.utils.general import read_json_from_data_dir

MWOZ_23_DATA_FILE: str = "mw23_data.json"


def _normalize_slot_name(slot_name: str) -> SlotName:
    fixes: Dict[str, SlotName] = {
        'booking-day': "hotel-area",
        'hotel-day': "hotel-book day",
        'hotel-people': "hotel-book people",
        'hotel-price': "hotel-pricerange",
        'restaurant-day': "restaurant-book day",
        'restaurant-people': "restaurant-book people",
        'restaurant-price': "restaurant-pricerange",
        'restaurant-time': "restaurant-book time",
        'taxi-arrive': "taxi-arriveby",
        'taxi-depart': "taxi-departure",
        'taxi-dest': "taxi-destination",
        'taxi-leave': "taxi-leaveat",
        'train-dest': "train-destination",
        'train-people': "train-book people"
    }
    # default to current value
    return fixes.get(slot_name, slot_name)


def get_coreference_annotations(mwoz_23_data_file: str = MWOZ_23_DATA_FILE) -> Dict[
    str, Dict[int, Dict[SlotName, Dict[str, str]]]]:
    result: Dict[str, Dict[int, Any]] = defaultdict(lambda: defaultdict(dict))

    data = read_json_from_data_dir(mwoz_23_data_file)
    for dial_id, dialogue_struct in tqdm(data.items(), desc="reading MultiWOZ 2.3 co-reference annotations"):
        for i, log in enumerate(dialogue_struct['log']):
            # [(0,), (1, 2), (3, 4) ...]
            turn_id: int = (i + 1) // 2
            if 'coreference' in log:
                for domain_intent, coreferences in log['coreference'].items():
                    for coreference in coreferences:
                        domain, _ = domain_intent.lower().split('-')
                        # e.g. Day, 'same day', 'saturday', (int), (str)
                        slot_str, coref_phrase, slot_value, _, _ = coreference
                        full_slot_name: SlotName = _normalize_slot_name(f"{domain}-{slot_str.lower()}")
                        result[dial_id][turn_id][full_slot_name] = {slot_value: coref_phrase}
    return result


if __name__ == '__main__':
    corefs = get_coreference_annotations()
    print(corefs)
