import argparse
import os
import sys
from typing import List, Callable

import numpy as np
import wandb
from refpydst.data_types import Turn, RetrieverFinetuneRunConfig
from sentence_transformers import SentenceTransformer, models, InputExample
from sentence_transformers.losses import OnlineContrastiveLoss
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from refpydst.retriever.code.data_management import MWDataset, save_embeddings, get_state_transformation_by_type, \
    StateTransformationFunction
from refpydst.retriever.code.data_management import get_string_transformation_by_type
from refpydst.retriever.code.embed_based_retriever import EmbeddingRetriever
from refpydst.retriever.code.index_based_retriever import IndexRetriever
from refpydst.retriever.code.pretrained_embed_index import embed_everything
from refpydst.retriever.code.retriever_evaluation import evaluate_retriever_on_dataset
from refpydst.retriever.code.st_evaluator import RetrievalEvaluator
from refpydst.utils.general import read_json, get_output_dir_full_path, REFPYDST_OUTPUTS_DIR, read_json_from_data_dir, \
    WANDB_ENTITY, WANDB_PROJECT


class MWContrastiveDataloader:
    """
    Constrastive Learning Data Loader w/ hard-negative sampling, from:

    @article{hu2022context,
      title={In-Context Learning for Few-Shot Dialogue State Tracking},
      author={Hu, Yushi and Lee, Chia-Hsuan and Xie, Tianbao and Yu, Tao and Smith, Noah A and Ostendorf, Mari},
      journal={arXiv preprint arXiv:2203.08568},
      year={2022}
    }
    """

    def __init__(self, f1_set: MWDataset, pretrained_retriever: IndexRetriever):
        """

        :param f1_set:
        :param pretrained_retriever:
        """
        self.f1_set = f1_set
        self.pretrained_retriever = pretrained_retriever

    def hard_negative_sampling(self, topk=10, top_range=100):
        sentences1 = []
        sentences2 = []
        scores = []

        # do hard negative sampling
        for ind in tqdm(range(self.f1_set.n_turns)):

            # find nearest neighbors given by pre-trained retriever
            this_label = self.f1_set.turn_labels[ind]
            nearest_labels = self.pretrained_retriever.label_to_nearest_labels(
                this_label, k=top_range + 1)[:-1]  # to exclude itself
            nearest_args = [self.f1_set.turn_labels.index(
                l) for l in nearest_labels]

            # topk and bottomk nearest f1 score examples, as hard examples
            similarities = self.f1_set.similarity_matrix[ind][nearest_args]
            sorted_args = similarities.argsort()

            chosen_positive_args = list(sorted_args[-topk:])
            chosen_negative_args = list(sorted_args[:topk])

            chosen_positive_args = np.array(nearest_args)[chosen_positive_args]
            chosen_negative_args = np.array(nearest_args)[chosen_negative_args]

            for chosen_arg in chosen_positive_args:
                sentences1.append(self.f1_set.turn_utts[ind])
                sentences2.append(self.f1_set.turn_utts[chosen_arg])
                scores.append(1)

            for chosen_arg in chosen_negative_args:
                sentences1.append(self.f1_set.turn_utts[ind])
                sentences2.append(self.f1_set.turn_utts[chosen_arg])
                scores.append(0)

        return sentences1, sentences2, scores

    def generate_eval_examples(self, topk=5, top_range=100):
        # add topk closest, furthest, and n_random random indices
        sentences1, sentences2, scores = self.hard_negative_sampling(
            topk=topk, top_range=top_range)
        scores = [float(s) for s in scores]
        return sentences1, sentences2, scores

    def generate_train_examples(self, topk=5, top_range=100):
        sentences1, sentences2, scores = self.generate_eval_examples(
            topk=topk, top_range=top_range)
        n_samples = len(sentences1)
        return [InputExample(texts=[sentences1[i], sentences2[i]], label=scores[i])
                for i in range(n_samples)]


def main(train_fn: str, dev_fn: str, test_fn: str, output_dir: str, pretrained_index_root: str = None,
         pretrained_model_full_name: str = 'sentence-transformers/all-mpnet-base-v2', num_epochs: int = 15,
         top_k: int = 10, top_range: int = 200,
         pooling_mode: str = None, f_beta: float = 1.0, log_wandb_freq: int = 100,
         str_transformation_type: str = "default", state_transformation_type: str = "default", **kwargs):
    wandb.config = dict(locals())
    train_set: List[Turn] = read_json_from_data_dir(train_fn)

    # prepare the retriever model
    word_embedding_model: models.Transformer = models.Transformer(pretrained_model_full_name, max_seq_length=512)
    pooling_model: models.Pooling = models.Pooling(
        word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
        pooling_mode=pooling_mode,
    )

    # embed everything to initialize a pre-trained index:
    if not pretrained_index_root:
        pretrained_index_root = get_output_dir_full_path(os.path.join("retriever/pretrained_index", pretrained_model_full_name))
    if not os.path.exists(os.path.join(pretrained_index_root, "mw21_train.npy")):
        embed_everything(model_name=pretrained_model_full_name,
                         output_dir=pretrained_index_root)

    # prepare pretrained retreiver for fine-tuning
    pretrained_train_retriever = IndexRetriever(
        datasets=[train_set],
        embedding_filenames=[
            f"{pretrained_index_root}/mw21_train.npy"
        ],
        search_index_filename=f"{pretrained_index_root}/mw21_train.npy",
        sampling_method="pre_assigned",
    )

    # Choose transformation (how each turn will be represented as a string for retriever training)
    string_transformation: Callable[[Turn], str] = get_string_transformation_by_type(str_transformation_type)
    state_transformation: StateTransformationFunction = get_state_transformation_by_type(state_transformation_type)

    # Preparing dataset
    f1_train_set = MWDataset(train_fn, beta=f_beta, string_transformation=string_transformation,
                             state_transformation=state_transformation)

    # Dataloader
    mw_train_loader = MWContrastiveDataloader(f1_train_set, pretrained_train_retriever)

    # add special tokens and resize
    tokens = ["[USER]", "[SYS]", "[CONTEXT]"]
    word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cuda:0")

    # prepare training dataloaders
    all_train_samples = mw_train_loader.generate_train_examples(topk=top_k, top_range=top_range)
    train_dataloader = DataLoader(all_train_samples, shuffle=True, batch_size=24)
    print(f"number of batches {len(train_dataloader)}")

    evaluator: RetrievalEvaluator = RetrievalEvaluator(train_fn=train_fn, dev_fn=dev_fn, index_set=f1_train_set,
                                                       string_transformation=string_transformation)

    # Training. Loss is constructed base on loss type argument
    train_loss: nn.Module = OnlineContrastiveLoss(model=model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=100,
              evaluator=evaluator, evaluation_steps=(len(train_dataloader) // 300 + 1) * 100,
              output_path=output_dir)

    # load best model
    model = SentenceTransformer(output_dir, device="cuda:0")

    # Note: previously this would embed all train set items, even those not in the training set. However this would risk
    # later use of this retriever and its indices with data it wasn't trained on that should be outside of its selection
    # pool. For now, not permitting this, and only saving the embeddings for the training set. If needed we can add an
    # explicit argument for the dataset to load and embed.
    save_embeddings(model, f1_train_set, os.path.join(output_dir, "train_index.npy"))

    test_set: List[Turn] = read_json_from_data_dir(test_fn)

    model.save(output_dir)
    retriever: EmbeddingRetriever = EmbeddingRetriever(
        datasets=[train_set],
        model_path=output_dir,
        search_index_filename=os.path.join(output_dir, "train_index.npy"),
        sampling_method="pre_assigned",
        string_transformation=string_transformation
    )

    # save the retriever as an artifact
    artifact: wandb.Artifact = wandb.Artifact(wandb.run.name, type="model")
    artifact.add_dir(output_dir)
    wandb.log_artifact(artifact)

    print("Now evaluating retriever ...")
    turn_sv, turn_s, dial_sv, dial_s = evaluate_retriever_on_dataset(test_set, retriever)
    wandb.log({
        "test_top_5_turn_slot_value_f_score": turn_sv,
        "test_top_5_turn_slot_name_f_score": turn_s,
        "test_top_5_hist_slot_value_f_score": dial_sv,
        "test_top_5_hist_slot_name_f_score": dial_s,
    })


if __name__ == '__main__':
    # input arguments
    if os.path.exists(sys.argv[1]):
        run_file: str = sys.argv[1]
        # arguments are input from a configuration file if the first argument to the program is a valid file
        args: RetrieverFinetuneRunConfig = read_json(run_file)
        if 'output_dir' not in args:
            args['output_dir'] = get_output_dir_full_path(run_file.replace('.json', ''))
        if 'run_name' not in args:
            args['run_name'] = args['output_dir'].replace(os.environ.get(REFPYDST_OUTPUTS_DIR, "outputs"), "").replace(
                '/', '-')
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_fn', type=str, required=True,
                            help="training data file (few-shot or full shot)")  # e.g. "../../data/mw21_10p_train_v3.json"
        parser.add_argument('--dev_fn', type=str, default="data/mw24_100p_dev.json",
                            help="dev data file (few-shot or full shot)")  # e.g. "../../data/mw21_10p_dev_v3.json"
        parser.add_argument('--test_fn', type=str, required=True,
                            help="test_fn data file (few-shot or full shot)")  # e.g. "../../data/mw21_10p_dev_v3.json"
        parser.add_argument('--output_dir', type=str, required=True,
                            help="sentence transformer save path")  # e.g. mw21_10p_v3
        parser.add_argument('--pretrained_index_dir', type=str,
                            default="retriever/pretrained_index/",
                            help="directory of pretrained embeddings")
        parser.add_argument('--pretrained_model', dest="pretrained_model_full_name", type=str,
                            default='sentence-transformers/all-mpnet-base-v2', help="embedding model to finetune with")
        parser.add_argument('--num_epochs', type=int, default=15)
        parser.add_argument('--top_k', type=int, default=10)
        parser.add_argument('--top_range', type=int, default=200)
        args = vars(parser.parse_args())
    default_run_name: str = args['output_dir'].replace("../expts/", "").replace('/', '-')
    default_run_group: str = default_run_name.rsplit('-', maxsplit=1)[0]
    wandb_entity: str = os.environ.get(WANDB_ENTITY, "kingb12")
    wandb_project: str = os.environ.get(WANDB_PROJECT, "refpydst")
    wandb.init(project=wandb_project, entity=wandb_entity, group=args.get("run_group", default_run_group),
               name=args.get("run_name", default_run_name))
    main(**args)
