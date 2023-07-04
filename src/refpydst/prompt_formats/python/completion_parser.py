"""
Classes and functions for parsing python completions to dialogue states
"""
import copy
import dataclasses
import logging
import pprint
import re
from typing import Union, Literal

import dacite
from dacite import from_dict
from refpydst.data_types import MultiWOZDict

from refpydst.prompt_formats.python.demo import normalize_to_domains_and_slots, SLOT_NAME_REVERSE_REPLACEMENTS
from refpydst.prompt_formats.python.python_classes import BeliefState, DialogueAgent, Taxi, PriceRange, Area, \
    DayOfWeek, Restaurant, AttractionType, Attraction, Train, HotelType, Option, Hotel

DomainBeliefState = Union[Hotel, Train, Attraction, Restaurant, Taxi]


@dataclasses.dataclass
class ParserBeliefState(BeliefState):
    """
    Class for parsing belief states. Implements methods included in the prompt, and modifies set-attribute to have
    upsert semantics as opposed to overwriting, yielding a complete dialogue state for the turn.
    """

    @staticmethod
    def from_dict(data):
        # we allow state to be invalid after parse, so we also need to allow here
        return from_dict(ParserBeliefState, data, config=dacite.Config(check_types=False))

    def update(self, data):
        pass  # these are used in some no-op demonstrations, not the actual updates

    def to_mwoz_dict(self):
        result = {}
        for domain in ["hotel", "train", "attraction", "restaurant", "taxi"]:
            if getattr(self, domain) is not None:
                domain_state = {k: v for k, v in dataclasses.asdict(getattr(self, domain)).items() if v is not None}
                for slot_name, slot_value in domain_state.items():
                    slot_name = SLOT_NAME_REVERSE_REPLACEMENTS.get(slot_name, slot_name.replace("_", " "))
                    result[f"{domain}-{slot_name}"] = str(slot_value)  # int's are strs in the evaluation dict
        return result

    def __setattr__(self, key: str, domain_obj: DomainBeliefState):
        """
        This hack essentially lets us translate an assignment operation into an update operation. Completions come in
        the form:

        agent.state.train = Train(destination="london", ...)

        In a normal program, one would of course expect this to write-over the current value of agent.state.train
        completely. Since the right-hand-side is for completion deltas from a dialogue turn, and we want to track the
        accumulating dialogue state across turns, this hack essentially translates the above to:

        agent.state.train.update(Train(destination="london", ...) with Dict.update semantics.
        """
        current: DomainBeliefState = getattr(self, key)
        if type(domain_obj) == ParserAgent:
            return
        if current is not None:
            # for each field in the value we're setting, check that the value we'd be writing isn't None. If it is,
            # ignore it to avoid over-writing. Deletions are handled separately with the `del` operator.
            for f in dataclasses.fields(domain_obj):
                new_value = getattr(domain_obj, f.name)
                # some exceptionally wrong completions will set an incorrect type that causes a stack overflow.
                # Generally we want to allow the code to execute as completed by the LM but in this case, don't enter
                # or the state can't be computed.
                # Example: "hotel = agent.find_hotel(agent.state.hotel)"
                # implicitly sets the name attribute to the current state Hotel object, causing infinite recursion when
                # parsing to a dictionary. Instead, don't set
                if new_value is not None and not isinstance(new_value, type(current)):
                    # we only should set primitive types in MultiWOZ, but some rare completions set dictionaries,
                    # including the state itself. In MultiWOZ, these are wrong, but we don't want the parser to be
                    # fixing such wrong completions from the LM. This also future proofs for hierarchical state
                    # definitions in other datasets, where such setting may be appropriate
                    setattr(current, f.name, copy.deepcopy(new_value))
        else:
            super().__setattr__(key, domain_obj)


def parser_belief_state_from_mwoz_dict(mwoz_dict: MultiWOZDict) -> "ParserBeliefState":
    """
    Given a MultiWOZDict in normalized form, parse it to a ParserBeliefState (e.g. for resolving references)

    :param mwoz_dict: the MultiWOZ normalized form of a belief state representation
    :return: corresponding representation for ParserBeliefState
    """
    normalized_state = normalize_to_domains_and_slots(mwoz_dict)
    return ParserBeliefState.from_dict(normalized_state)


class ParserAgent(DialogueAgent):
    """
    Class used in parsing belief states. Implements the function calls in DialogueAgent to return the appropriate
    Dataclass, which can be used to modify one of the attributes in a BeliefState.
    """
    state: ParserBeliefState

    def __init__(self):
        self.state = ParserBeliefState()

    def find_hotel(self, name: str = None, price_range: PriceRange = None, type: HotelType = None,
                   parking: Option = None, book_number_of_days: int = None, book_day: DayOfWeek = None,
                   book_people: int = None, area: Area = None, stars: Union[int, Literal["dontcare"]] = None,
                   internet: Option = None, **kwargs) -> Hotel:
        return Hotel(name=name, price_range=price_range, type=type, parking=parking,
                     book_number_of_days=book_number_of_days, book_day=book_day, book_people=book_people, area=area,
                     stars=stars, internet=internet)

    def find_train(self, destination: str = None, leave_from: str = None, day: DayOfWeek = None,
                   book_people: int = None, depart_time: str = None, arrive_by_time: str = None, **kwargs) -> Train:
        return Train(destination=destination, leave_from=leave_from, day=day, book_people=book_people,
                     depart_time=depart_time, arrive_by_time=arrive_by_time)

    def find_attraction(self, name: str = None, area: Area = None, type: AttractionType = None, **kwargs) -> Attraction:
        return Attraction(name=name, area=area, type=type)

    def find_restaurant(self, name: str = None, food_type: str = None, price_range: PriceRange = None,
                        area: Area = None, book_time: str = None, book_day: DayOfWeek = None,
                        book_people: int = None, **kwargs) -> Restaurant:
        return Restaurant(name=name, food_type=food_type, price_range=price_range, area=area, book_time=book_time,
                          book_day=book_day, book_people=book_people)

    def find_taxi(self, destination: str = None, leave_from: str = None, depart_time: str = None,
                  arrive_by_time: str = None, **kwargs) -> Taxi:
        return Taxi(destination=destination, leave_from=leave_from, depart_time=depart_time,
                    arrive_by_time=arrive_by_time)


def replace_state_references(full_statement: str) -> str:
    # isolated for testing
    return re.sub(pattern=r"([^.])state\.", repl=r"\1agent.state.", string=full_statement)


def parse_python_completion(python_completion: str, state: Union[MultiWOZDict, ParserBeliefState] = None,
                            exceptions_are_empty: bool = True, **kwargs) -> MultiWOZDict:
    """
    Parses a python completion to a complete dialogue state for the turn, y_t.

    :param python_completion: the dialogue state update in python function-call form
    :param state: the existing dialogue state (y_{t-1})
    :param exceptions_are_empty: If an exception is encountered (un-parseable completion), treat this as containing no
      update to the dialogue state.
    :param kwargs: Not used, but included as other parsers use different arguments.
    :return: the updated dialogue state y_t.
    """
    try:
        if not type(state) == ParserBeliefState:
            state = parser_belief_state_from_mwoz_dict(state)
        full_statement = "agent.state." + python_completion.strip()
        full_statement = replace_state_references(full_statement)
        agent = ParserAgent()
        agent.state = state or ParserBeliefState()
        lines = []
        for line in full_statement.splitlines():
            if line.strip().startswith("del "):
                # Our evaluation treats deletions as assignment to a special keyword, modify to do this:
                # e.g. del agent.state.hotel.type -> agent.data.hotel.type = "[DELETE]"
                line = line.replace("del ", "") + " = \"[DELETE]\""
            lines.append(line.strip())
        statements = "\n".join(lines).strip()
        exec(statements)
        return agent.state.to_mwoz_dict()
    except Exception as e:
        print(f"got exception when parsing: {pprint.pformat(e)}")
        logging.warning(e)
        if not exceptions_are_empty:
            raise e
        return agent.state.to_mwoz_dict()
