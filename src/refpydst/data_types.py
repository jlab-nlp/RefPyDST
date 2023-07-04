from typing import Dict, Union, TypedDict, Optional, Literal

_TurnValue = Union[str, int, float]
Turn = Dict[str, _TurnValue]

# SlotNames are complete names including a domain and a slot, separated by a dash. e.g. "hotel-area"
SlotName = Literal["attraction-area", "attraction-name", "attraction-type", "bus-day", "bus-departure",
                   "bus-destination", "bus-leaveat", "hospital-department", "hotel-area", "hotel-book day",
                   "hotel-book people", "hotel-book stay", "hotel-internet", "hotel-name", "hotel-parking",
                   "hotel-pricerange", "hotel-stars", "hotel-type", "restaurant-area", "restaurant-book day",
                   "restaurant-book people", "restaurant-book time", "restaurant-food", "restaurant-name",
                   "restaurant-pricerange", "taxi-arriveby", "taxi-departure", "taxi-destination", "taxi-leaveat",
                   "train-arriveby", "train-book people", "train-day", "train-departure", "train-destination",
                   "train-leaveat"]

SlotValue = Union[str, int]  # Most commonly strings, but occasionally integers

# MultiWOZ Dict is the dictionary format for slot values as provided in the dataset. It is flattened, and denotes a
# dictionary that can be immediately evaluated using exact-match based metrics on keys and values. keys are in
# domain-slot form e.g. {"hotel-area": "centre", ...}
MultiWOZDict = Dict[SlotName, SlotValue]


# A completion parser takes a string completion and returns a MultiWOZ normalized dictionary of slot pairs
class CompletionParser:
    def __call__(self, completion: str, context: MultiWOZDict, **kwargs) -> MultiWOZDict:
        pass


class RetrieverFinetuneRunConfig(TypedDict):
    """
    A configuration for a retrieval training experiment run, as defined by any JSON file in the runs/ directory
    """
    train_fn: Optional[str]
    test_fn: Optional[str]
    pretrained_index_dir: Optional[str]
    pretrained_model: Optional[str]
    num_epochs: Optional[int]
    run_group: Optional[str]
    run_name: Optional[str]
    output_dir: Optional[str]
    top_k: Optional[int]
    top_range: Optional[int]


ExampleListDecoderType = Literal["top_k", "min_distinct", "max_emb_distance"]


class ExampleListDecoderConfig(TypedDict):
    """
    Configuration for using some instance of AbstractExampleListDecoder to for decoding a list of examples from a
    scoring retriever. Default would be TopKDecoder()
    """
    decoder_type: ExampleListDecoderType
    num_distinct_demonstrations: Optional[int]
    discount_factor: Optional[float]
    from_n_possible: Optional[int]


class CodexDecodingConfig(TypedDict):
    method: Optional[Literal["greedy", "top_p", "mutual_info"]]
    top_p: Optional[float]
    max_mi_candidates: Optional[int]
    null_prompt_format: Optional[str]
    min_null_probability: float
    min_token_null_probability: Optional[float]
    stop_token: Optional[str]
    canonical_completions: Optional[bool]
    null_prompt_weight: Optional[float]


class CodexPromptingRunConfig(TypedDict):
    """
    A configuration for an experiment run, as defined by any JSON file in the runs/ directory
    """
    train_fn: Optional[str]
    retriever_dir: Optional[str]
    test_fn: Optional[str]
    prompt_format: Optional[str]
    run_group: Optional[str]
    run_name: Optional[str]
    output_dir: Optional[str]
    num_distinct_demonstrations: Optional[int]
    retriever_args: Optional[Dict[str, str]]
    decoder_config: Optional[ExampleListDecoderConfig]
    lm_decoding_config: Optional[CodexDecodingConfig]


SlotValuesByDomain = Dict[str, Dict[str, SlotValue]]
