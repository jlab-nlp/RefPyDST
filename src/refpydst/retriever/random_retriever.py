"""
A retriever which samples random examples from a training set (random.choices)
"""
import random
from typing import List

from refpydst.data_types import Turn

from refpydst.retriever.abstract_example_retriever import ExampleRetriever
from refpydst.retriever.abstract_example_set_decoder import AbstractExampleListDecoder


class RandomExampleRetriever(ExampleRetriever):
    def __init__(self, datasets: List[List[Turn]], **kwargs) -> None:
        self.data_items: List[Turn] = []
        for dataset in datasets:
            self.data_items.extend(dataset)

    def item_to_best_examples(self, turn: Turn, k: int = 10, decoder: AbstractExampleListDecoder = None) -> \
            List[Turn]:
        return random.choices(self.data_items, k=k)
