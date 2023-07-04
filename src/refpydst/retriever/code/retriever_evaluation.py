import argparse
from statistics import mean
from typing import List

from refpydst.data_types import Turn
from tqdm import tqdm

from refpydst.utils.general import read_json_from_data_dir


def compute_prf(gold: List[str], pred: List[str], beta: float = 1) -> float:
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP) != 0 else 0
        recall = TP / float(TP+FN) if (TP+FN) != 0 else 0
        f_beta_numerator: float = ((1 + beta) * precision * recall)
        f_beta_denominator: float = (beta**2 * precision) + recall
        f_beta: float = float(f_beta_numerator / f_beta_denominator) if f_beta_denominator != 0 else 0.0
    else:
        if len(pred) == 0:
            precision, recall, f_beta, count = 1, 1, 1, 1
        else:
            precision, recall, f_beta, count = 0, 0, 0, 1
    return float(f_beta)


def multival_to_single(belief):
    return [f"{'-'.join(sv.split('-')[:2])}-{(sv.split('-')[-1]).split('|')[0]}" for sv in belief]


# mean of slot similarity and value similarity
def compute_sv_sim(gold, pred, onescore=True, beta: float = 1):

    if type(gold) == dict:
        gold = [f"{k}-{v}" for k, v in gold.items()]
    if type(pred) == dict:
        pred = [f"{k}-{v}" for k, v in pred.items()]

    gold = multival_to_single(gold)
    pred = multival_to_single(pred)

    value_sim = compute_prf(gold, pred, beta=beta)

    gold = ['-'.join(g.split('-')[:2]) for g in gold]
    pred = ['-'.join(g.split('-')[:2]) for g in pred]
    slot_sim = compute_prf(gold, pred)

    if onescore:
        return value_sim + slot_sim - 1
    else:
        return value_sim, slot_sim


def evaluate_single_query_ex(turn, retriever, beta: float = 1):
    examples = retriever.item_to_best_examples(turn)

    query_turn_sv = turn['turn_slot_values']
    query_sv = turn['slot_values']

    turn_value_sims = []
    turn_slot_sims = []
    all_value_sims = []
    all_slot_sims = []

    for ex in examples:
        this_turn_sv = ex['turn_slot_values']
        this_sv = ex['slot_values']

        turn_value_sim, turn_slot_sim = compute_sv_sim(
            query_turn_sv, this_turn_sv, onescore=False, beta=beta)
        all_value_sim, all_slot_sim = compute_sv_sim(query_sv, this_sv, onescore=False, beta=beta)

        turn_value_sims.append(turn_value_sim)
        turn_slot_sims.append(turn_slot_sim)
        all_value_sims.append(all_value_sim)
        all_slot_sims.append(all_slot_sim)

    return mean(turn_value_sims), mean(turn_slot_sims), mean(all_value_sims), mean(all_slot_sims)


def evaluate_retriever_on_dataset(dataset, retriever, beta: float = 1):
    turn_value_sims = []
    turn_slot_sims = []
    all_value_sims = []
    all_slot_sims = []

    for ds in tqdm(dataset):
        turn_value_sim, turn_slot_sim, all_value_sim, all_slot_sim = evaluate_single_query_ex(
            ds, retriever, beta=beta)
        turn_value_sims.append(turn_value_sim)
        turn_slot_sims.append(turn_slot_sim)
        all_value_sims.append(all_value_sim)
        all_slot_sims.append(all_slot_sim)

    return mean(turn_value_sims), mean(turn_slot_sims), mean(all_value_sims), mean(all_slot_sims)


if __name__ == '__main__':
    from refpydst.retriever.code.embed_based_retriever import EmbeddingRetriever

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fn', type=str, required=True, help="training data file (few-shot or full shot)")  # e.g. "../../data/mw21_10p_train_v3.json"
    parser.add_argument('--retriever_dir', type=str, required=True,
                        help="sentence transformer saved path")  # "./retriever/expts/mw21_10p_v3_0304_400_20"
    parser.add_argument('--test_fn', type=str, required=True, help="test data file (few-shot or full-shot)")
    parser.add_argument('--beta', type=float, required=False, default=1.0,
                        help="beta parameter for f-beta score. Default is 1. If supplied, recall is considered beta "
                             "times more important than precision")

    args = parser.parse_args()
    train_set: List[Turn] = read_json_from_data_dir(args.train_fn)
    test_set: List[Turn] = read_json_from_data_dir(args.test_fn)
    retriever = EmbeddingRetriever(datasets=[train_set],
                                   model_path=args.retriever_dir,
                                   search_index_filename=f"{args.retriever_dir}/train_index.npy",
                                   sampling_method="pre_assigned")
    print("Now evaluating retriever against top-5 retrieved examples: avg. F1 of (turn slots-and-values, turn slots, "
          "full hist slots-and-values, full hist slots")
    turn_sv, turn_s, dial_sv, dial_s = evaluate_retriever_on_dataset(test_set, retriever, beta=args.beta)
    print(turn_sv, turn_s, dial_sv, dial_s)