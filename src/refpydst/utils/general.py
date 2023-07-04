import json
import logging
import os
from numbers import Number
from pathlib import Path
from typing import Optional, Any, Iterable, Iterator, Dict, Union

import numpy as np
import numpy.typing as npt
import torch
from refpydst.data_types import Turn
from torch import Tensor

# environment variable name for passing huggingface access token. Not natively supported in HF libraries, need to get
# from os.environ and use in relevant calls.
HF_API_TOKEN: str = "HF_API_TOKEN"
# environment variable names for file management
REFPYDST_DATA_DIR: str = "REFPYDST_DATA_DIR"
REFPYDST_OUTPUTS_DIR: str = "REFPYDST_OUTPUTS_DIR"

# environment variable names for wandb management
WANDB_ENTITY: str = "WANDB_ENTITY"
WANDB_PROJECT: str = "WANDB_PROJECT"

def check_argument(assertion: Any, message: Optional[str]) -> None:
    """
    Checks the assertion is true, and raises a Value error if it is false, with the provided message
    :param assertion: assertion to check (can be a truthy, such as a non-empty vs. empty list)
    :param message: message to provide if assertion fails
    :return: None
    """
    if not assertion:
        raise ValueError(message)


def read_json(file_name: Union[str, Path]) -> Any:
    with open(file_name, 'r') as f:
        return json.load(f)


def read_json_from_data_dir(file_name: Union[str, Path]) -> Any:
    try:
        if not os.path.isabs(file_name) and REFPYDST_DATA_DIR in os.environ:
            # relative path and we have a data dir set, read from that path
            file_name = os.path.join(os.environ[REFPYDST_DATA_DIR], file_name)
        return read_json(file_name)
    except FileNotFoundError as e:
        logging.error(f"data file not found. Use absolute paths or set {REFPYDST_DATA_DIR} environment variable "
                      f"correctly. ({REFPYDST_DATA_DIR}={os.environ.get(REFPYDST_DATA_DIR)})")
        raise e


def get_output_dir_full_path(file_path: Union[str, Path]) -> Union[str, Path]:
    """
    If the file_path is not absolute, return the file path such that it is rooted in the configured outputs directory

    :param file_path: relative or absolute path. relative paths will be re-mapped to relative to REFPYDST_OUTPUT_DIR
      or outputs/ by default. Absolute paths are preserved.
    :return: file path with any modifications
    """
    if not os.path.isabs(file_path) and REFPYDST_OUTPUTS_DIR in os.environ:
        return os.path.join(os.environ[REFPYDST_OUTPUTS_DIR], file_path)
    return file_path


def np_top_k(arr: npt.ArrayLike, k: int, sort_result: bool = True, pad_value: Any = None) -> npt.NDArray:
    """
    A numpy alias for top-k like behavior via np.partition (more efficient than sorting first)

    :param arr: npt.ArrayLike, anything you can pass to np.partition
    :param k: number of top values to return
    :return:
    """
    if len(arr) < k:
        arr = np.pad(arr, (0, k - len(arr)), constant_values=pad_value, mode='constant')
    result = np.partition(arr, -k)[-k:]
    if sort_result:
        result = np.sort(result)[::-1]
    return result


def round_robin(iterables: Iterable[Iterable[Any]]) -> Iterator[Any]:
    iterators = [iter(l) if not isinstance(l, Iterator) else l for l in iterables]
    yieldable: bool = True
    while yieldable:
        yieldable = False
        for it in iterators:
            item = next(it, None)
            if item:
                yieldable = True
                yield item


def subtract_dict(x: Dict[Any, Number], y: Dict[Any, Number], default_value: float = 0) -> Dict[Any, Number]:
    return {k: x[k] - y.get(k, default_value) for k in x}


def cosine_similarity_matrix(x: Tensor, y: Tensor, eps: float=1e-8) -> Tensor:
    """
    Given input matrices x (N, D) and y (M, D), return an N x M matrix of the cosine similarities
    between all rows
    :param x: a matrix where every row is a vector of size D, to compute cosine similarities to rows in y
    :param y: a matrix where every row is a vector of size D, to compute cosine similarities to rows in x
    :return: an N x M matrix z where z[i, j] is the cosine similarity between x[i] and y[j]
    """
    # https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
    x_n, y_n = x.norm(dim=1)[:, None], y.norm(dim=1)[:, None]
    a_norm = x / torch.clamp(x_n, min=eps)
    b_norm = y / torch.clamp(y_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def print_dialogue(turn: Turn, last_k=0) -> None:
    for (user, sys) in zip(turn['dialog']['usr'][-last_k:], turn['dialog']['sys'][-last_k:]):
        print(f"sys: {sys}")
        print(f"usr: {user}")

    print("\nGold Update:")
    print(turn['turn_slot_values'])
