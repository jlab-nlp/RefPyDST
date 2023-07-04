"""
Much of this file adapted from the YushiHu/IC-DST
@article{hu2022context,
      title={In-Context Learning for Few-Shot Dialogue State Tracking},
      author={Hu, Yushi and Lee, Chia-Hsuan and Xie, Tianbao and Yu, Tao and Smith, Noah A and Ostendorf, Mari},
      journal={arXiv preprint arXiv:2203.08568},
      year={2022}
}
"""
import logging
from typing import Dict, List

import numpy.typing as npt
import wandb
from sentence_transformers import models
from sentence_transformers.evaluation import SentenceEvaluator

from refpydst.retriever.code.data_management import MWDataset
from refpydst.retriever.code.embed_based_retriever import EmbeddingRetriever
from refpydst.retriever.code.retriever_evaluation import evaluate_retriever_on_dataset
from refpydst.utils.general import read_json_from_data_dir

logger = logging.getLogger(__name__)


class RetrievalEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.
    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.
    :param batch_size: Batch size used to compute embeddings
    :param show_progress_bar: If true, prints a progress bar
    """
    train_fn: str
    dev_fn: str
    index_set: MWDataset

    def __init__(self, train_fn: str, dev_fn: str, index_set: MWDataset, batch_size: int = 32,
                 show_progress_bar: bool = False, string_transformation=None):
        self.train_fn = train_fn
        self.dev_fn = dev_fn
        self.index_set = index_set
        self.batch_size = batch_size
        self.string_transformation = string_transformation
        if show_progress_bar is None:
            show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        logger.info("Evaluating")
        scores = self.compute_metrics(model)

        wandb.log({
            "epoch": epoch,
            "step": steps,
            **{"dev_" + k: v for k, v in scores.items()}
        })
        # main score is the average F1 between slot-name only and slot-names-and-values
        return scores['top_5_turn_slot_value_f_score'] + scores['top_5_turn_slot_name_f_score']

    def compute_metrics(self, model: models.Transformer) -> Dict[str, float]:
        # retriever to evaluation
        train_set = read_json_from_data_dir(self.train_fn)
        dev_set = read_json_from_data_dir(self.dev_fn)

        embeddings: Dict[str, List[npt.NDArray]] = {
            k: v for k, v in zip(
                self.index_set.turn_labels,  # key is a dialogue id and turn id combined as a string
                model.encode(self.index_set.turn_utts, convert_to_numpy=True)  # value is a singleton list w/ embedding
            )
        }

        retriever = EmbeddingRetriever(datasets=[train_set],
                                       model_path="",
                                       model=model,
                                       search_embeddings=embeddings,
                                       sampling_method="pre_assigned",
                                       string_transformation=self.string_transformation)

        turn_sv, turn_s, dial_sv, dial_s = evaluate_retriever_on_dataset(dev_set, retriever)
        return {'top_5_turn_slot_value_f_score': turn_sv, "top_5_turn_slot_name_f_score": turn_s,
                "top_5_hist_slot_value_f_score": dial_sv, "top_5_hist_slot_name_f_score": dial_s}
