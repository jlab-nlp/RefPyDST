import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from refpydst.utils.general import read_json_from_data_dir


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# function for embedding one string
def embed_single_sentence(sentence, tokenizer: AutoTokenizer, model: AutoModel, cls=False):
    device = model.device
    # Sentences we want sentence embeddings for
    sentences = [sentence]

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(input_ids, attention_mask)

    # Perform pooling
    sentence_embeddings = None

    if cls:
        sentence_embeddings = model_output[0][:, 0, :]
    else:
        sentence_embeddings = mean_pooling(model_output, attention_mask)

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


def read_MW_dataset(mw_json_fn):
    # only care domain in test
    DOMAINS = ['hotel', 'restaurant', 'attraction', 'taxi', 'train']

    data = read_json_from_data_dir(mw_json_fn)

    dial_dict = {}

    for turn in data:
        # filter the domains that not belongs to the test domain
        if not set(turn["domains"]).issubset(set(DOMAINS)):
            continue

        # update dialogue history
        sys_utt = turn["dialog"]['sys'][-1]
        usr_utt = turn["dialog"]['usr'][-1]

        if sys_utt == 'none':
            sys_utt = ''
        if usr_utt == 'none':
            usr_utt = ''

        history = f"[system] {sys_utt} [user] {usr_utt}"

        # store the history in dictionary
        name = f"{turn['ID']}_turn_{turn['turn_id']}"
        dial_dict[name] = history

    return dial_dict


def store_embed(input_dataset, output_filename, forward_fn):
    outputs = {}
    with torch.no_grad():
        for k, v in tqdm(input_dataset.items()):
            outputs[k] = forward_fn(v).detach().cpu().numpy()
    np.save(output_filename, outputs)
    return


def embed_everything(model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                     output_dir: str = f"outputs/retriever/pretrained_index/"):
    # path to save indexes and results
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0")

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)

    mw_train = read_MW_dataset("mw21_100p_train.json")
    print("Finish reading data")

    def embed_sentence_with_this_model(sentence):
        return embed_single_sentence(sentence, model=model, tokenizer=tokenizer, cls=False)

    store_embed(mw_train, f"{output_dir}/mw21_train.npy",
                embed_sentence_with_this_model)
    print("Finish Embedding data")


if __name__ == '__main__':
    embed_everything()
