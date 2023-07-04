from typing import Dict

BASE_CONVERSION_DICT = {"leaveat": "depart_time", "arriveby": "arrive_by_time",
                        "book_stay": "book_number_of_days",
                        "food": "food_type"}


def promptify_slot_names(prompt: str, conversion_dict: Dict[str, str] = None, reverse=False) -> str:
    conversion_dict = conversion_dict or BASE_CONVERSION_DICT
    reverse_conversion_dict = {v: k for k, v in conversion_dict.items()}
    used_dict = reverse_conversion_dict if reverse else conversion_dict

    for k, v in used_dict.items():
        prompt = prompt.replace(k, v)
    return prompt
