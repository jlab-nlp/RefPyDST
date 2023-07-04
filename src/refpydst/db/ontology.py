import itertools
import json
import logging
import re
from collections import defaultdict
from typing import Dict, List, TypedDict, Optional, Any, get_args, Set, Tuple

from fuzzywuzzy import process, fuzz
from num2words import num2words
from refpydst.data_types import SlotName, SlotValue

from refpydst.resources import _read_resource

# Categorical slots can plausibly be checked at parse time by a real system, where names
# would require a DB round-trip
CATEGORICAL_SLOT_VALUE_TYPES: List[str] = ["area", "stars", "type", "parking", "pricerange", "internet",
                                           "book day", "day", "department", "book people",
                                           "book stay"]

class SchemaSlotDefinition(TypedDict):
    name: str
    description: Optional[str]
    possible_values: Optional[List[str]]
    is_categorical: bool


Schema = Dict[SlotName, SchemaSlotDefinition]

TIME_SLOT_SUFFIXES: List[str] = ['leaveat', 'arriveby', 'book time']

# two named entities forms are considered as referring to the same canonical object if adding one of the prefixes
# or suffixes maps to that entity in the DB ontology
ENTITY_NAME_PREFIXES = ['the ']
ENTITY_NAME_SUFFIXES = [" hotel", " restaurant", ' cinema', ' guest house',
                        " theatre", " airport", " street", ' gallery', ' museum', ' train station']


def insert_space(token, text):
    """
    This function was adapted from the code for the paper "In Context Learning for Dialogue State Tracking", as
    originally published here: https://github.com/Yushi-Hu/IC-DST. Cite their article as:

    @article{hu2022context,
      title={In-Context Learning for Few-Shot Dialogue State Tracking},
      author={Hu, Yushi and Lee, Chia-Hsuan and Xie, Tianbao and Yu, Tao and Smith, Noah A and Ostendorf, Mari},
      journal={arXiv preprint arXiv:2203.08568},
      year={2022}
    }

    I believe it is also derived from the original MultiWOZ repository: https://github.com/budzianowski/multiwoz
    """
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text


def normalize(text: str) -> str:
    """
    This function was adapted from the code for the paper "In Context Learning for Dialogue State Tracking", as
    originally published here: https://github.com/Yushi-Hu/IC-DST. Cite their article as:

    @article{hu2022context,
      title={In-Context Learning for Few-Shot Dialogue State Tracking},
      author={Hu, Yushi and Lee, Chia-Hsuan and Xie, Tianbao and Yu, Tao and Smith, Noah A and Ostendorf, Mari},
      journal={arXiv preprint arXiv:2203.08568},
      year={2022}
    }

    I believe it is also derived from the original MultiWOZ repository: https://github.com/budzianowski/multiwoz
    """
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)
    text = re.sub(r"guesthouse", "guest house", text)

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\"\<>@\(\)]', '', text)  # remove

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insert_space(token, text)

    # insert white space for 's
    text = insert_space('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    # remove the added spaces before s
    text = re.sub(' s ', 's ', text)
    text = re.sub(' s$', 's', text)

    value_replacement = {'center': 'centre',
                         'caffe uno': 'cafe uno',
                         'caffee uno': 'cafe uno',
                         'christs college': 'christ college',
                         'cambridge belfy': 'cambridge belfry',
                         'churchill college': 'churchills college',
                         'sat': 'saturday',
                         'saint johns chop shop house': 'saint johns chop house',
                         'good luck chinese food takeaway': 'good luck',
                         'asian': 'asian oriental',
                         'gallery at 12': 'gallery at 12 a high street'}

    if text in value_replacement:
        text = value_replacement[text]
    return text


def parse_schema(schema_json: List[Dict[str, Any]]) -> Schema:
    """
    Parsing the contents of `schema.json` into something indexed by slot name
    :param schema_json: JSON loaded `schema.json` contents,
        see https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.2/schema.json
    :return: slot-names to slot definitions
    """
    schema: Schema = {}
    for service in schema_json:
        for slot in service['slots']:
            schema[slot['name'].lower()] = slot
    return schema


class Ontology:
    known_values: Dict[SlotName, Set[SlotValue]]
    schema: Schema
    min_fuzzy_match: int
    found_matches: Optional[Dict[SlotName, Dict[str, str]]]
    valid_slots: Set[SlotName]

    def __init__(self, known_values: Dict[SlotName, Set[SlotValue]],
                 schema: Schema,
                 min_fuzzy_match: int = 95,
                 track_matches: bool = False):
        self.known_values = known_values
        self.min_fuzzy_match = min_fuzzy_match
        self.found_matches = defaultdict(dict) if track_matches else None
        self.schema = schema
        self.valid_slots = set(get_args(SlotName))

    # separating for testing
    @staticmethod
    def is_valid_time(value: SlotValue) -> bool:
        return bool(re.match(r"^([0-1]?[0-9]|2[0-4]):[0-5][0-9]$", value))

    @staticmethod
    def _per_digit_num2words(token: str) -> str:
        if len(token) > 1 and token.isnumeric():
            return ' '.join(num2words(digit) for digit in token)
        else:
            return num2words(token)

    def is_categorical(self, slot_name: SlotName) -> bool:
        schema_slot_name = self._get_schema_slot_name(slot_name)
        return schema_slot_name in self.schema and self.schema[schema_slot_name].get('is_categorical')

    def is_name(self, slot_name: SlotName) -> bool:
        return slot_name.split('-')[1] == 'name'

    # separating for readability and testing
    @staticmethod
    def numeral_aliases(value: SlotValue) -> Set[SlotValue]:
        aliases = set()
        tokens = value.split()  # default to white-space tokenization for handling numerals
        numeric_indices: List[int] = [i for (i, token) in enumerate(tokens) if token.isnumeric()]
        # this is exhaustive, but should work generally
        for subset_size in range(len(numeric_indices) + 1):
            for combination in itertools.combinations(numeric_indices, subset_size):
                aliases.add(' '.join(num2words(token) if i in combination else token for (i, token) in
                                     enumerate(tokens)))
                # consider multi-digit tokens as having both full-number and per-digit aliases:
                # restaurant 17 = restaurant seventeen AND restaurant one seven
                aliases.add(' '.join(Ontology._per_digit_num2words(token) if i in combination else token
                                     for (i, token) in enumerate(tokens)))
        return aliases

    @staticmethod
    def get_acceptable_aliases(value: SlotValue) -> List[SlotValue]:
        aliases = {value}
        # first, consider possible truncations of the given value (removing prefix or suffix)
        for prefix in ENTITY_NAME_PREFIXES:
            accepted_alternates = []
            if value.startswith(prefix):
                # add JUST truncating the prefix
                aliases.add(value[len(prefix):])
                # track alternates in case we need to drop prefix AND suffix
                accepted_alternates.append(value[len(prefix):])
            for suffix in ENTITY_NAME_SUFFIXES:
                if value.endswith(suffix):
                    # add JUST truncating the suffix
                    aliases.add(value[:-len(suffix)])
                    # add truncating both, if we've truncated a prefix
                    aliases.update([alt[:-len(suffix)] for alt in accepted_alternates])
        # consider all combinations of adding and removing a prefix/suffix. In a test and code we'll ensure we aren't
        # creating transformations for a single value that match 2+ distinct entities (since these should be aliases for
        # just one entity
        for alias in list(aliases):
            for prefix in ENTITY_NAME_PREFIXES:
                if not alias.startswith(prefix):
                    # prefix not present. add prefix
                    aliases.add(prefix + alias)
                    # also check if we can add suffixes WITH this prefix added
                    for suffix in ENTITY_NAME_SUFFIXES:
                        if not alias.endswith(suffix):
                            aliases.add(prefix + alias + suffix)
            # for each alias, also consider only suffixes
            for suffix in ENTITY_NAME_SUFFIXES:
                if not alias.endswith(suffix):
                    aliases.add(alias + suffix)

        # Finally, for all aliases, consider aliases for numerals to words e.g. 'restaurant 2 2' -> 'restaurant two two'
        numeral_aliases = set()
        for alias in aliases:
            numeral_aliases.update(Ontology.numeral_aliases(alias))
        aliases.update(numeral_aliases)
        return list(aliases)

    def get_canonical(self, full_slot_name: SlotName, value: SlotValue) -> Optional[SlotValue]:
        """
        For a given full slot name (e.g. 'hotel-name'), convert the given value into its canonical form. The canonical
        form for a slot value (e.g. name) is the form defined in the original database for entity it references. E.g:
        surface forms 'the acorn guest house', 'acorn guest house', 'the acorn guesthouse' all de-reference to
        canonical form 'acorn guest house', as defined in db/multiwoz/hotel_db.json

        :param full_slot_name: the complete slot name (domain, slot, separated by dash, lowercased). e.g. 'hotel-name'
        :param value: the value to convert. Does not need to be a name, could be a category or timestamp
            (e.g. we handle '5:14' -> '05:14')
        :return: canonical form of the value for the given slot, or None if there is not one (which implies the value
           is not in the ontology).
        """
        if full_slot_name not in self.valid_slots:
            logging.warning(f"seeking a canonical value for an unknown slot_name={full_slot_name}, slot_value={value}")
            return None
        domain, short_slot_name = full_slot_name.split('-')
        if full_slot_name in self.known_values:
            # direct match: value is already canonical
            if value in self.known_values[full_slot_name]:
                return value
            else:
                # Add acceptable prefixes and suffixes such that we hopefully find an exact match. A test verifies these
                # uniquely identify an object, instead of two aliases for the same value yielding different db objects
                aliases = self.get_acceptable_aliases(value)
                for alias in aliases:
                    if alias in self.known_values[full_slot_name]:
                        # this is the canonical alias which matches an actual DB entity name
                        if self.found_matches is not None:
                            self.found_matches[full_slot_name][value] = alias
                        return alias
                # No direct matches. Finally, attempt a fuzzy match (could be a mispelling, e.g. 'pizza hut fenditton'
                # vs. 'pizza hut fen ditton'
                fuzzy_matches: List[Tuple[str, str, int]] = []
                for alias in aliases:
                    # fuzz.ratio does NOT account for partial phrase matches, which is preferred, since these can
                    # have surprisingly high scores mapping from generic to specific, e.g:
                    # 'archaeology' -> 'museum of science and archaeology' is pretty high. Since we consider so many
                    # aliases, we want to be sure
                    best_match, best_score = process.extractOne(alias, self.known_values[full_slot_name],
                                                                scorer=fuzz.ratio)
                    if best_score >= self.min_fuzzy_match:
                        fuzzy_matches.append((best_match, alias, best_score))
                unique_matches: Set[str] = set(match for match, _, _ in fuzzy_matches)
                if len(unique_matches) > 1:
                    print(f"Warning: a had aliases yielding two distinct fuzzy matches. Consider increasing "
                          f"min_fuzz_value: {fuzzy_matches}")
                    return None
                else:
                    # all the same, just get the first
                    if fuzzy_matches:
                        match, alias, score = fuzzy_matches[0]
                        if self.found_matches is not None:
                            self.found_matches[full_slot_name][value] = match
                        return match
                    return None

        elif short_slot_name in TIME_SLOT_SUFFIXES:
            # convert 9:00 -> 09:00
            if ':' in value and len(value) < 5:
                value = '0' + value
            # then: verify it is actually a time-stamp in (00:00 -> 23:59)
            return value if self.is_valid_time(value) or value == 'dontcare' else None
        else:
            raise ValueError(f"unexpected slot name {full_slot_name}")

    def is_in_ontology(self, slot_name: SlotName, value: SlotValue) -> bool:
        try:
            return self.get_canonical(slot_name, value) is not None
        except ValueError as e:
            return False

    @staticmethod
    def _get_schema_slot_name(slot_name: str) -> str:
        # handle key differences between schema.json and the rest of the system
        return slot_name.replace(' ', '')

    @staticmethod
    def create_ontology(min_fuzzy_match: int = 90, track_matches: bool = False) -> "Ontology":
        known_values: Dict[SlotName, Set[SlotValue]] = {}

        # read schema
        schema = parse_schema(json.loads(_read_resource("db/multiwoz/schema.json")))

        # read database files
        domain_dbs = {
            domain: json.loads(_read_resource(f"db/multiwoz/{domain}_db.json"))
            for domain in ('attraction', 'bus', 'hospital', 'hotel', 'police', 'restaurant', 'taxi', 'train')
        }

        # iterate over the slots we care about (defined in Literal SlotName, retrievable via get_args)
        time_slots = []
        location_slots = []
        for full_slot_name in get_args(SlotName):
            # for a few slots, the dev set and our references add a space, e.g. 'hotel-book day' vs. 'hotel-bookday'
            schema_slot_name = Ontology._get_schema_slot_name(full_slot_name)
            schema_domain, schema_slot = schema_slot_name.split('-')  # hotel-area => hotel, area
            if schema[schema_slot_name]['is_categorical']:
                # categorical slots define their possible values in the schema
                known_values[full_slot_name] = set(schema[schema_slot_name]['possible_values'] + ['dontcare'])
            elif schema_slot in ('leaveat', 'arriveby', 'booktime'):
                # these are time-based slots, we'll need to validate with functions vs. possible values
                time_slots.append(full_slot_name)
            elif schema_slot in ('departure', 'destination'):
                # these are location/entity based slots, derived from other slots, fill in later.
                location_slots.append(full_slot_name)
            else:
                # non-categorical ones do not (e.g. hotel names), but we can reference all values present in the database
                # for these
                domain_db = domain_dbs[schema_domain]
                known_values[full_slot_name] = set([normalize(entity[schema_slot]) for entity in domain_db] +
                                                   ['dontcare'])
        locations: Set[str] = {'dontcare'}
        for slot_name in ('attraction-name', 'hospital-name', 'hotel-name', 'police-name', 'restaurant-name'):
            locations.update(known_values.get(slot_name, []))

        # some locations exist only as referenced in departure/destination locations of trains, busses
        for domain in ('bus', 'train'):
            for journey in domain_dbs[domain]:
                locations.add(journey['destination'])
                locations.add(journey['departure'])
        for slot_name in location_slots:
            known_values[slot_name] = locations

        return Ontology(known_values, schema=schema, min_fuzzy_match=min_fuzzy_match, track_matches=track_matches)

    def is_valid_slot(self, slot_name: str) -> bool:
        return slot_name is not None and slot_name in self.valid_slots

