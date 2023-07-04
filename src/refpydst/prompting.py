from typing import List, Dict

from refpydst.data_types import Turn, CompletionParser, MultiWOZDict

import prompt_formats.python.demo as python_demo
from refpydst.prompt_formats.python.completion_parser import parse_python_completion
from refpydst.prompt_formats.utils import promptify_slot_names
from refpydst.resources import _read_resource
from refpydst.utils.dialogue_state import compute_delta
from refpydst.utils.general import read_json_from_data_dir
from refpydst.utils.sql import slot_values_to_seq_sql, sql_pred_parse

IC_DST: str = "IC-DST"
IC_DST_NULL: str = "IC-DST-NULL"
PYTHON_PROMPT: str = "python-prompt"
PYTHON_PROMPT_NULL: str = "python-prompt-NULL"
PROMPT_VARIANTS: List[str] = [IC_DST, IC_DST_NULL, PYTHON_PROMPT, PYTHON_PROMPT_NULL]


STOP_SEQUENCES: Dict[str, List[str]] = {
    PYTHON_PROMPT: ["\n\n", "#", "print("],
    IC_DST: ['--', '\n', ';', '#']
}


def default_sql_completion_parser(completion: str, _: MultiWOZDict, **kwargs) -> MultiWOZDict:
    # convert back the sql completion result
    completion = promptify_slot_names(completion, reverse=True)
    return sql_pred_parse(completion)


def get_completion_parser(prompt_format: str) -> CompletionParser:
    if prompt_format == PYTHON_PROMPT:
        return parse_python_completion
    else:
        return default_sql_completion_parser


class PromptGenerator:
    """
    A class handling the creation of prompts for various experiments
    """
    preambles: Dict[str, str]

    def __init__(self):
        ic_dst_table_prompt: str = _read_resource("prompt_formats/ic_dst/table.sql")
        self.preambles = {
            IC_DST: ic_dst_table_prompt,
            IC_DST_NULL: ic_dst_table_prompt,
            PYTHON_PROMPT: _read_resource("prompt_formats/python/python_classes.py"),
            PYTHON_PROMPT_NULL: _read_resource("prompt_formats/python/python_classes.py"),
        }

    def get_prompt(self, data_item, examples, given_context=None, n_examples=None, prompt_format: str = None):
        """
        You can try different prompt in here.
        """
        # Note the IC-DST text-to-sql prompts are all prefixed with "IC-DST":
        if not prompt_format or prompt_format in (IC_DST, IC_DST_NULL):
            reverse_x_and_y, use_null_data_item = False, False
            if prompt_format == IC_DST_NULL:
                reverse_x_and_y, use_null_data_item = True, True
            return self.get_icdst_prompt(data_item, examples, given_context=given_context, n_examples=n_examples,
                                         reverse_x_and_y=reverse_x_and_y, use_null_data_item=use_null_data_item)
        elif prompt_format in (PYTHON_PROMPT, PYTHON_PROMPT_NULL):
            reverse_x_and_y, use_null_data_item = (True, True) \
                if prompt_format == PYTHON_PROMPT_NULL else (False, False)
            detailed_state_string: bool = prompt_format == PYTHON_PROMPT
            return self.get_python_prompt(data_item, examples, given_context=given_context, n_examples=n_examples,
                                          reverse_x_and_y=reverse_x_and_y, use_null_data_item=use_null_data_item,
                                          detailed_state_string=detailed_state_string)
        else:
            raise ValueError(f"Unsupported prompt format: {prompt_format}")

    @staticmethod
    def get_canonical_completion(slot_values: MultiWOZDict, context_slot_values, turn: Turn,
                                 prompt_format: str = None):
        """
        For a given true value y or prediction y_hat, generate a string that the LM could have completed given the
        relevant prompt in order to produce y/y_hat when parsed
        """
        # Note the IC-DST text-to-sql prompts are all prefixed with "IC-DST":
        slot_delta = compute_delta(context_slot_values, slot_values)
        if not prompt_format or prompt_format.startswith(IC_DST):
            # chop off the end, as we complete it in the prompts
            return f"{promptify_slot_names(slot_values_to_seq_sql(slot_delta))}".replace("SELECT * FROM", "")
        elif prompt_format in (PYTHON_PROMPT, PYTHON_PROMPT_NULL):
            last_sys_utt = turn['dialog']['sys'][-1]
            if last_sys_utt == 'none':
                last_sys_utt = ''
            user_string = python_demo.get_user_string(turn['dialog']['usr'][-1])
            _, update_string = python_demo.get_python_statements(context_slot_values, slot_values,
                                                                 turn_strings=[last_sys_utt, user_string])
            if update_string.startswith("agent.state."):
                return update_string.replace("agent.state.", "", 1)
            return update_string
        else:
            raise ValueError(f"Unsupported prompt format: {prompt_format}")

    def get_icdst_prompt(self, data_item, examples, given_context=None, n_examples=None, add_context: bool = True,
                         reverse_x_and_y: bool = False, use_null_data_item: bool = False):
        """
        Prompt as originally proposed in the IC-DST paper
        """
        table_prompt = self.preambles[IC_DST]

        question_item = data_item

        prompt_text = f"{promptify_slot_names(table_prompt)}\n"

        max_n_examples = len(examples)
        if n_examples is not None:
            max_n_examples = n_examples

        # in case for zero-shot learning
        if max_n_examples > 0:
            for example_id, example in enumerate(examples[-max_n_examples:]):
                turn_text = f"Example #{example_id + 1}\n"
                turn_input_text = ""
                # remove multiple choice in last slot values
                if add_context:
                    last_slot_values = {s: v.split('|')[0] for s, v in example['last_slot_values'].items()}
                    turn_input_text += f"[context] {promptify_slot_names(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"

                last_sys_utt = example['dialog']['sys'][-1]
                if last_sys_utt == 'none':
                    last_sys_utt = ''
                turn_input_text += f"[system] {last_sys_utt}\n"
                turn_input_text += f"Q: [user] {example['dialog']['usr'][-1]}\n"

                turn_output_text = f"SQL: {promptify_slot_names(slot_values_to_seq_sql(example['turn_slot_values']))};\n"

                # set the text for this turn, depending on order preference
                turn_text += (turn_input_text + turn_output_text) if not reverse_x_and_y else \
                    (turn_output_text + turn_input_text)
                prompt_text += turn_text + "\n\n"

        prompt_text += f"Example #{max_n_examples + 1}\n"
        if given_context is None:
            last_slot_values = {s: v.split(
                '|')[0] for s, v in question_item['last_slot_values'].items()}
        else:
            last_slot_values = given_context
        test_example_text: str = ""
        if add_context:
            test_example_text += f"[context] {promptify_slot_names(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"

        last_sys_utt = question_item['dialog']['sys'][-1]
        if last_sys_utt == 'none':
            last_sys_utt = ''
        test_example_text += f"[system] {last_sys_utt}\n"
        test_example_text += f"Q: [user] {question_item['dialog']['usr'][-1]}\n"
        test_example_text += "SQL: SELECT * FROM"

        if not use_null_data_item:
            prompt_text += test_example_text
        else:
            # use a null input (note for now we have not chosen a leading null X -> Y input)
            prompt_text += ("SQL: SELECT * FROM" if reverse_x_and_y else "")
        return prompt_text

    def get_python_prompt(self, data_item, examples, given_context=None, n_examples: int = None,
                          reverse_x_and_y: bool = False, use_null_data_item: bool = False,
                          detailed_state_string: bool = False) -> str:
        lines: List[str] = []
        max_n_examples: int = n_examples is not None and n_examples or len(examples)

        # in case for zero-shot learning
        if max_n_examples > 0:
            for example_id, example in enumerate(examples[-max_n_examples:]):
                lines.append(f"# Example {example_id + 1}")
                turn_inputs, turn_outputs = [], []

                # remove multiple choice in last slot values
                last_slot_values = {s: v.split('|')[0] for s, v in example['last_slot_values'].items()}

                last_sys_utt = example['dialog']['sys'][-1]
                if last_sys_utt == 'none':
                    last_sys_utt = ''
                user_string = python_demo.get_user_string(example['dialog']['usr'][-1])
                state_string, update_string = python_demo.get_python_statements(last_slot_values, example['slot_values'],
                                                                                turn_strings=[last_sys_utt, user_string],
                                                                                detailed_state_string=detailed_state_string)
                turn_inputs.append(state_string)
                if last_sys_utt:
                    turn_inputs.append(python_demo.get_system_string(last_sys_utt))
                turn_inputs.append(user_string)
                turn_outputs.extend([s.strip() for s in update_string.split("\n")])
                if not reverse_x_and_y:
                    lines.extend(turn_inputs)
                    lines.extend(turn_outputs)
                else:
                    lines.extend(turn_outputs)
                    lines.extend(turn_inputs)
                lines.append("")

        lines.append(f"# Example {max_n_examples + 1}")
        if given_context is None:
            last_slot_values = {s: v.split(
                '|')[0] for s, v in data_item['last_slot_values'].items()}
        else:
            last_slot_values = given_context
        last_sys_utt = data_item['dialog']['sys'][-1]
        if last_sys_utt == 'none':
            last_sys_utt = ''
        user_string = python_demo.get_user_string(data_item['dialog']['usr'][-1])
        state_string, _ = python_demo.get_python_statements(last_slot_values, {},
                                                            turn_strings=[last_sys_utt, user_string],
                                                            detailed_state_string=detailed_state_string)
        if not use_null_data_item:
            lines.append(state_string)
            if last_sys_utt:
                lines.append(python_demo.get_system_string(last_sys_utt))
            lines.append(user_string)
        else:
            pass  # default adds our null input at end
        prompt_text = self.preambles[PYTHON_PROMPT] + "    " + "\n    ".join(lines) + "\n    agent.state."
        return prompt_text


if __name__ == '__main__':
    pg = PromptGenerator()
    data = read_json_from_data_dir("mw24_10p_dev.json")
    train_data = read_json_from_data_dir("mw21_5p_train_v2.json")
    DEMONSTRATION_EXAMPLES = [
        {
            "dialog": {
                "sys": ["i have booked that for you , is there anything else i can help with ?"],
                "usr": [
                    "thank you , i will need a taxi from clare college to the restaurant . i need to get there by reservation time .",
                ]
            },
            "turn_slot_values": {
                "taxi-departure": "clare college",
                "taxi-destination": "restaurant alimentum",
                "taxi-arriveby": "6:15",
            },
            "last_slot_values": {"restaurant-name": "restaurant alimentum", "restaurant-food": "modern european",
                                 "restaurant-book time": "6:15", "restaurant-book people": "2",
                                 "restaurant-book day": "tuesday"},
            "slot_values": {"restaurant-name": "restaurant alimentum", "restaurant-food": "modern european",
                                 "restaurant-book time": "6:15", "restaurant-book people": "2",
                                 "restaurant-book day": "tuesday","taxi-departure": "clare college",
                "taxi-destination": "restaurant alimentum",
                "taxi-arriveby": "6:15",}
        }
    ]
    prompt = pg.get_prompt(data[306], train_data[1:11], n_examples=10, given_context=data[306]['last_slot_values'],
                           prompt_format=PYTHON_PROMPT)
    print(prompt)
