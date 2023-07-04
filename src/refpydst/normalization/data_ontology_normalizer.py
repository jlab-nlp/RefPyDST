from collections import defaultdict, Counter
from typing import List, Dict

from refpydst.data_types import Turn, SlotName, MultiWOZDict, SlotValue
from tqdm import tqdm

from normalization.abstract_normalizer import AbstractNormalizer
from refpydst.db.ontology import Ontology
from refpydst.utils.general import read_json
from resources import read_json_resource


class DataOntologyNormalizer(AbstractNormalizer):

    # S" -> C
    ontology: Ontology
    supervised_set: List[Turn]
    # C -> S'
    canonical_to_surface: Dict[SlotName, str]

    def __init__(self, ontology: Ontology,
                 supervised_set: List[Turn] = None,
                 counts_from_ontology_file: str = None,
                 per_occurrence_in_ontology_file: int = 10) -> None:
        """
        Creates a data ontology normalizer. This combines two main components:
        1) An ontology definition, constructed from the system DB definition, which can be used for mapping from
           a surface form S" to a canonical form in the database/schema C (S" -> C)
        2) A set of counts of gold label 'surface forms' from annotations. These would usually be un-necessary in DST,
           but are required for effective evaluation in toy problems like MultiWOZ, in which JGA is computed against
           exact match of predicted strings for each slot. Each canonical form C becomes associated with a single most
           likely annotated surface form S', derived from data, where available. This comes in two forms:
             - supervised_set: the training set for this run, if applicable. Given these, we count surface forms in
               labels to derive the most likely S' for a given C.
             - counts_from_ontology_file: often, MultiWOZ dataset providers and pipelines construct a list of known
               surface forms for each slot name. To fairly evaluate when compared to prior methods, we also take
               this list when available, and assume we've seen each string within it K times (no direct dialogue
               observation), where K=per_occurrence_in_ontology_file (default=10)

        :param ontology: the ontology constructed from Schema/DB files, maps S" -> C
        :param supervised_set: training data for this run, used to choose the most likely annotated surface form C -> S'
        :param counts_from_ontology_file: given a path to a resource or JSON file mapping slot names to a list of slot
               values, counts present surface forms K=per_occurrence_in_ontology_file times each. While we consider
               ontology to refer to the DB/schema structure, this is commonly in a file named ontology.json in other
               works
        :param per_occurrence_in_ontology_file: number of times to count each ontology file surface form
        """
        super().__init__()
        self.ontology = ontology
        self.canonical_to_surface: Dict[SlotValue, Counter[SlotValue]] = defaultdict(lambda: Counter())
        if supervised_set:
            for turn in tqdm(supervised_set, desc="mapping supervised_set surface forms..."):
                for i, (slot, values) in enumerate(turn['slot_values'].items()):
                    for value in values.split("|"):
                        canonical = ontology.get_canonical(slot, value)
                        if canonical is not None:
                            self.canonical_to_surface[canonical][value] += 1
        if counts_from_ontology_file:
            self.counts_from_ontology_file(counts_from_ontology_file, per_occurence=per_occurrence_in_ontology_file)

    def get_most_common_surface_form(self, slot_name: SlotName, slot_value: SlotValue, keep_counting: bool = False) -> \
            SlotValue:
        """
        For a given slot and value, return the most common surface form for the value referenced by slot_value. Most
        common is determined by labels in the given 'supervised_set' (in a practical setting, this includes train & dev
        sets but not test).
        :param slot_name:
        :param slot_value:
        :param keep_counting:
        :return:
        """
        canonical_form: SlotValue = self.ontology.get_canonical(slot_name, slot_value)
        if canonical_form is None:
            return None
        elif canonical_form not in self.canonical_to_surface:
            # in the training set, this surface form was never found. Return as 'most common' and count if permitted
            if keep_counting:
                self.canonical_to_surface[canonical_form][slot_value] += 1
            return slot_value
        else:
            # we have seen this canonical form before, get most common observed surface form for it
            # canonical_form -> Counter -> [(form, count)] -> (form, count) -> form
            return self.canonical_to_surface[canonical_form].most_common(1)[0][0]

    def normalize(self, raw_parse: MultiWOZDict, **kwargs) -> MultiWOZDict:
        """
        Given a 'raw' parse, normalize the slot values to best match the surface forms expected in the evaluation set.
        These surface forms are determined by the supervised data given when instantiating the normalizer, such that an
        authentically few-shot normalization process can be used by appropriately scoping the surface form count data.

        :param raw_parse: a dictionary of slots to un-normalized values
        :return: a dictionary of slots to normalized values. Normalization process may omit existing slots.
        """
        normalized: MultiWOZDict = {}
        for slot_name, slot_value in raw_parse.items():
            if not self.ontology.is_valid_slot(slot_name):
                continue
            if type(slot_value) == str:
                slot_value = slot_value.split("|")[0]
            normalized_form: SlotValue = self.get_most_common_surface_form(slot_name, slot_value)
            if normalized_form:
                normalized[slot_name] = normalized_form
            elif not self.ontology.is_categorical(slot_name) and not self.ontology.is_name(slot_name):
                # preserve prediction values for non-categorical slots, even if they are wrong
                normalized[slot_name] = slot_value
        return normalized

    def _counts_from_json(self, json_path: str) -> None:
        # useful in tests, to skip time consuming counting process
        data = read_json_resource(json_path)
        for canonical_form, counts in data.items():
            self.canonical_to_surface[canonical_form].update(counts)

    def counts_from_ontology_file(self, ontology_file: str, per_occurence: int = 10) -> None:
        """
        The ontology.json file is often released in data preparation with different MultiWOZ versions.
        This reads in those surface forms, and adds a count of 1 for each

        :param ontology_file: path to ontology.json (e.g. see MultiWOZ 2.4 data repo)
        :return: None
        """
        try:
            # one of the files we've included in this package
            ontology_data: Dict[SlotName, List[str]] = read_json_resource(ontology_file)
        except BaseException as e:
            # possibly some other file
            ontology_data: Dict[SlotName, List[str]] = read_json(ontology_file)

        for slot_name, slot_value_strings in tqdm(ontology_data.items(),
                                                  desc=f"reading surface forms from ontology.json"):
            for slot_value_string in slot_value_strings:
                for slot_value in slot_value_string.split("|"):
                    canonical = self.ontology.get_canonical(slot_name, slot_value)
                    if canonical is not None:
                        self.canonical_to_surface[canonical][slot_value] += per_occurence
        return
