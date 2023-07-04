import random
from typing import Tuple, List, Iterator, Dict, Type, Callable, Union

import numpy as np
import torch
from numpy.typing import NDArray
from refpydst.data_types import Turn
from scipy.spatial import KDTree
from sentence_transformers import SentenceTransformer

from refpydst.retriever.abstract_example_retriever import ExampleRetriever
from refpydst.retriever.abstract_example_set_decoder import AbstractExampleListDecoder
from refpydst.retriever.code.data_management import get_string_transformation_by_type, data_item_to_string
from refpydst.retriever.decoders.top_k import TopKDecoder

TurnLabel = str


class Retriever:
    label_to_idx: Dict[str, int]

    def normalize(self, emb):
        return emb / np.linalg.norm(emb, axis=-1, keepdims=True)

    def __init__(self, emb_dict):

        # to query faster, stack all search embeddings and record keys
        self.emb_keys: List[str] = list(emb_dict.keys())
        self.label_to_idx = {k: i for i, k in enumerate(self.emb_keys)}
        emb_dim = emb_dict[self.emb_keys[0]].shape[-1]

        self.emb_values = np.zeros((len(self.emb_keys), emb_dim))
        for i, k in enumerate(self.emb_keys):
            self.emb_values[i] = emb_dict[k]

        # normalize for cosine distance (kdtree only support euclidean when p=2)
        self.emb_values = self.normalize(self.emb_values)
        self.kdtree = KDTree(self.emb_values)

    def iterate_nearest_dialogs(self, query_emb, k=5) -> Iterator[Tuple[str, float]]:
        query_emb = self.normalize(query_emb)
        i = 0
        fetch_size: int = k
        while i < len(self.emb_keys):
            scores, query_result = self.kdtree.query(query_emb, k=fetch_size, p=2)
            if query_result.shape == (1,):
                i += 1
                yield self.emb_keys[query_result.item()], scores.item()
            else:
                for item, score_item in zip(query_result.squeeze(0)[i:], scores.squeeze(0)[i:]):
                    i += 1
                    if item.item() >= len(self.emb_keys):
                        return  # stop iteration!
                    yield self.emb_keys[item.item()], score_item.item()
            fetch_size = min(2 * fetch_size, len(self.emb_keys))

    def topk_nearest_dialogs(self, query_emb, k=5):
        query_emb = self.normalize(query_emb)
        query_result = self.kdtree.query(query_emb, k=k, p=2)
        if k == 1:
            return [self.emb_keys[i] for i in query_result[1]]
        return [self.emb_keys[i] for i in query_result[1][0]]

    def topk_nearest_distinct_dialogs(self, query_emb, k=5):
        return self.topk_nearest_dialogs(query_emb, k=k)

    def random_retrieve(self, k=5):
        return random.sample(self.emb_keys, k)


class EmbeddingRetriever(ExampleRetriever):

    # sample selection
    def random_sample_selection_by_turn(self, embs, ratio=0.1):
        n_selected = int(ratio * len(embs))
        print(f"randomly select {ratio} of turns, i.e. {n_selected} turns")
        selected_keys = random.sample(list(embs), n_selected)
        return {k: v for k, v in embs.items() if k in selected_keys}

    def random_sample_selection_by_dialog(self, embs, ratio=0.1):
        dial_ids = set([turn_label.split('_')[0] for turn_label in embs.keys()])
        n_selected = int(len(dial_ids) * ratio)
        print(f"randomly select {ratio} of dialogs, i.e. {n_selected} dialogs")
        selected_dial_ids = random.sample(dial_ids, n_selected)
        return {k: v for k, v in embs.items() if k.split('_')[0] in selected_dial_ids}

    def pre_assigned_sample_selection(self, embs, examples):
        selected_dial_ids = set([dial['ID'] for dial in examples])
        return {k: v for k, v in embs.items() if k.split('_')[0] in selected_dial_ids}

    def __init__(self, datasets, model_path, search_index_filename: str = None, sampling_method="none", ratio=1.0,
                 model=None,
                 search_embeddings=None, full_history=False, retriever_type: Type[Retriever] = Retriever,
                 string_transformation: Union[str, Callable[[Turn], str]] = None, **kwargs):

        # data_items: list of datasets in this notebook. Please include datasets for both search and query
        # embedding_filenames: list of strings. embedding dictionary npy files. Should contain embeddings of the datasets. No need to be same
        # search_index:  string. a single npy filename, the embeddings of search candidates
        # sampling method: "random_by_turn", "random_by_dialog", "kmeans_cosine", "pre_assigned"
        # ratio: how much portion is selected

        def default_transformation(turn):
            return data_item_to_string(turn, full_history=full_history)

        if type(string_transformation) == str:
            # configs can also specify known functions by a string, e.g. 'default'
            self.string_transformation = get_string_transformation_by_type(string_transformation)
        else:
            self.string_transformation = string_transformation or default_transformation

        self.data_items = []
        for dataset in datasets:
            self.data_items += dataset

        # save all embeddings and dial_id_turn_id in a dictionary
        self.model = model

        if model is None:
            self.model = SentenceTransformer(model_path)

        # load the search index embeddings
        if search_embeddings is not None:
            self.search_embeddings = search_embeddings
        elif search_index_filename:
            self.search_embeddings = np.load(search_index_filename, allow_pickle=True).item()
        else:
            raise ValueError("unable to instantiate a retreiver without embeddings. Supply pre-loaded search_embeddings"
                             " or a search_index_filename")

        # sample selection of search index
        if sampling_method == "none":
            self.retriever = retriever_type(self.search_embeddings)
        elif sampling_method == 'random_by_dialog':
            self.retriever = retriever_type(self.random_sample_selection_by_dialog(self.search_embeddings, ratio=ratio))
        elif sampling_method == 'random_by_turn':
            self.retriever = retriever_type(self.random_sample_selection_by_turn(self.search_embeddings, ratio=ratio))
        elif sampling_method == 'pre_assigned':
            self.retriever = retriever_type(self.pre_assigned_sample_selection(self.search_embeddings, self.data_items))
        else:
            raise ValueError("selection method not supported")

    def data_item_to_embedding(self, data_item):
        with torch.no_grad():
            embed = self.model.encode(self.string_transformation(
                data_item), convert_to_numpy=True).reshape(1, -1)
        return embed

    def data_item_to_label(self, turn: Turn) -> TurnLabel:
        return f"{turn['ID']}_turn_{turn['turn_id']}"

    def label_to_data_item(self, label: TurnLabel) -> Turn:
        ID, _, turn_id = label.split('_')
        turn_id = int(turn_id)

        for d in self.data_items:
            if d['ID'] == ID and d['turn_id'] == turn_id:
                return d
        raise ValueError(f"label {label} not found. check data items input")

    def item_to_best_examples(self, data_item, k=5, decoder: AbstractExampleListDecoder = TopKDecoder()):
        # the nearest neighbor is at the end
        query = self.data_item_to_embedding(data_item)
        try:
            return decoder.select_k(k=k, examples=((self.label_to_data_item(turn_label), score) for turn_label, score in
                                                   self.retriever.iterate_nearest_dialogs(query, k=k)))
        except StopIteration as e:
            print("ran out of examples! unable to decode")
            raise e

    def random_examples(self, data_item, k=5):
        return [self.label_to_data_item(l)
                for l in self.retriever.random_retrieve(k=k)
                ]

    def get_turns_in_embedding_radius(self, turn: Turn, radius: float) -> List[Tuple[TurnLabel, float]]:
        embedding = self.retriever.normalize(self.data_item_to_embedding(turn))
        [t for t in self.retriever.iterate_nearest_dialogs(embedding, k=4)]
        # the maximum euclidean distance between any two unit vectors is 2
        self.retriever.kdtree.query_ball_point(embedding, r=radius, )

    def label_to_search_embedding(self, label: TurnLabel) -> NDArray:
        """
        For a known search turn (e.g. a retrieved example), get its embedding from it's label

        :param label: the string label used to identify turns when instantiating the retriever (emb_dict keys)
        :return: that initialized value, which is the search embedding
        """
        if label not in self.retriever.label_to_idx:
            raise KeyError(f"{label} not in search index")
        return self.retriever.emb_values[self.retriever.label_to_idx[label]]
