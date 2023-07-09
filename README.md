# RefPyDST: Diverse Retrieval-Augmented In-Context Learning for Dialogue State Tracking

This is the code release for the paper Diverse Retrieval-Augmented In-Context Learning for Dialogue State Tracking [[pdf]](https://arxiv.org/pdf/2307.01453.pdf)

Brendan King and Jeffrey Flanigan.

Findings of ACL 2023. Long Paper


```
@misc{king2023diverse,
      title={Diverse Retrieval-Augmented In-Context Learning for Dialogue State Tracking}, 
      author={Brendan King and Jeffrey Flanigan},
      year={2023},
      eprint={2307.01453},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Installation

The below steps successfully install and run the experiments on a CentOS 7 Linux host with
at least one GPU and `conda` available. You may need to adapt the steps as necessary for your environment.

##### Environment Setup

Set up an environment and install the specific python/torch versions used in this work. Other versions of each may also
work but are not tested.

```bash
mkdir refpydst && cd refpydst
conda create --prefix venv python=3.9.13
conda activate ./venv
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# Below was needed when building my docker image, but may not be needed for your install
conda install -c anaconda gxx_linux-64
pip install pyzmq
```

##### Install in edit mode

You can install this project and all dependencies in edit mode, or omit the `-e` flag for non-edit mode:

```bash
pip install -e .
```

These installation steps should be sufficient for reproducing results, please open an [Issue](https://github.com/kingb12/refpydst/issues) if not.

## Downloading Dataset and configuring Data Directory

We download and process data following the processing in [Yushi-Hu/IC-DST](https://github.com/Yushi-Hu/IC-DST.git):

```bash
cd data
python create_data.py --main_dir mwz21 --mwz_ver 2.1 --target_path mwz2.1  # for MultiWOZ 2.1
python create_data.py --main_dir mwz24 --mwz_ver 2.4 --target_path mwz2.4  # for MultiWOZ 2.4

```

Pre-processing creates train/dev/test splits for each zero and few-shot experiment in this work.

```bash
bash preprocess.sh
```

You can also sample your own as follows (ex: 5% training data with random seed=0)

```bash
python sample.py --input_fn mwz2.1/train_dials.json --target_fn mw21_5p_train_seed0.json --ratio 0.05 --seed 0
```

For co-reference analysis experiments, we use annotations provided by MultiWOZ 2.3:

```bash
bash download_mw23.sh
```

To efficiently analyze only dialogues that require co-reference in the dev set, we filter with this script:

```bash
python build_coref_only_dataset.py
```

which creates `mw24_coref_only_dials_dev.json` in the data folder.

### Configuring Data Directory

By default, all of these will download and create processed files in `./data`. Alternatively, you move this directory 
to a convenient location (e.g. something in a volume mount path) and point to it with an environment variable:

```bash
export REFPYDST_DATA_DIR="/absolute/path/to/data"
```

**When this environment variable is set, all dataset loading of relative paths will look from this root. Absolute paths
will always be loaded without modification**

## Re-running experiments

### Environment Variables

Reproducing our results requires access to OpenAI Codex (`code-davinci-002`). Other OpenAI engines are not tested. 
You'll need to set the following environment variables:

```bash
export OPENAI_API_KEY=<your_openai_api_key_with_codex_access>
export REFPYDST_DATA_DIR="`pwd`/data"  # default, or set as abs. path to where you want to store contents of data created above
export REFPYDST_OUTPUTS_DIR="`pwd`/outputs"  # default, or set as abs. path to where you want to save/load retrievers and output logs

# You may also want to set CUDA_VISIBLE_DEVICES to a single available GPU (can be small)
```

### Experiment Organization

Experiments are generally organized as follows:

1. A config file (json) specifies all the parameters for the experiment
2. A main file exists per experiment type, which can be called with `python [path/to/main_file].py [path/to/run/config].json`
   1. `src/refpydst/retriever/code/retriever_finetuning.py`: fine-tunes a sentence retriever given retriever arguments
       in config.
   2. `src/refpydst/run_codex_experiment.py`: runs an experiment for generating dialogue states with OpenAI Codex

Each few-shot experiment is repeated on 3 independent runs, so we've created utility scripts for running all experiments
in a folder. For example, to run all 1% few-shot experiments on MultiWOZ 2.4, you can do the following:

- Retriever Training: `bash retriever_expmt_folder_runner.sh runs/retriever/mw21_1p_train`
- Generation w/ Codex:  `bash codex_expmt_folder_runner.sh runs/codex/mw21_1p_train/python`
- Simulating MultiWOZ 2.1 Results (from 2.4 Codex run above): `bash mw21_simulation_expmt_folder_runner.sh runs/codex/mw21_1p_train/sim_mw21_python`

### Setting up `wandb`

This repo makes use of `wandb` for logging and storing of output artifacts (logs of completed runs, etc.). To override
the default project and entity, you can set these environment variables:

```bash
# default is mine: kingb12. You can find this in code and change as well
export WANDB_ENTITY=<your wandb account name or org-name/entity>
# default is refpydst for all runs so setting this here is a no-op. Re-name as needed
export WANDB_PROJECT="refpydst"
```

### Training Retrievers

You can train all few-shot retrievers as follows:

```bash
bash retriever_expmt_folder_runner.sh runs/retriever/mw21_1p_train/referred_states
bash retriever_expmt_folder_runner.sh runs/retriever/mw21_5p_train/referred_states
bash retriever_expmt_folder_runner.sh runs/retriever/mw21_10p_train/referred_states
bash retriever_expmt_folder_runner.sh runs/retriever/mw21_100p_train/referred_states
bash retriever_expmt_folder_runner.sh runs/retriever/mw21_5p_train/ic_dst  # for ablation
```

Or an individual run:

```bash
python src/refpydst/retriever/code/retriever_finetuning.py runs/retriever/mw21_1p_train/referred_states/split_v1.json
```

### Pre-trained Retrievers

We've made pre-trained retrievers from this repository available on Huggingface:

| Model Name                                      | Few-shot Setting | Run # | Link                                                                        |
|-------------------------------------------------|------------------|-------|-----------------------------------------------------------------------------|
| `Brendan/refpydst-1p-referredstates-split-v1`   | 1%               | 1     | [link](https://huggingface.co/Brendan/refpydst-1p-referredstates-split-v1)  |
| `Brendan/refpydst-1p-referredstates-split-v2`   | 1%               | 2     | [link](https://huggingface.co/Brendan/refpydst-1p-referredstates-split-v2)  |
| `Brendan/refpydst-1p-referredstates-split-v3`   | 1%               | 3     | [link](https://huggingface.co/Brendan/refpydst-1p-referredstates-split-v3)  |
| `Brendan/refpydst-5p-referredstates-split-v1`   | 5%               | 1     | [link](https://huggingface.co/Brendan/refpydst-5p-referredstates-split-v1)  |
| `Brendan/refpydst-5p-referredstates-split-v2`   | 5%               | 2     | [link](https://huggingface.co/Brendan/refpydst-5p-referredstates-split-v2)  |
| `Brendan/refpydst-5p-referredstates-split-v3`   | 5%               | 3     | [link](https://huggingface.co/Brendan/refpydst-5p-referredstates-split-v3)  |
| `Brendan/refpydst-10p-referredstates-split-v1`  | 10%              | 1     | [link](https://huggingface.co/Brendan/refpydst-10p-referredstates-split-v1) |
| `Brendan/refpydst-10p-referredstates-split-v2`  | 10%              | 2     | [link](https://huggingface.co/Brendan/refpydst-10p-referredstates-split-v2) |
| `Brendan/refpydst-10p-referredstates-split-v3`  | 10%              | 3     | [link](https://huggingface.co/Brendan/refpydst-10p-referredstates-split-v3) |
| `Brendan/refpydst-100p-referredstates-split-v3` | 100%             | 1     | [link](https://huggingface.co/Brendan/refpydst-100p-referredstates)         |
| `Brendan/refpydst-5p-icdst-split-v1`            | 5%               | 1     | [link](https://huggingface.co/Brendan/refpydst-5p-icdst-split-v1)           |
| `Brendan/refpydst-5p-icdst-split-v2`            | 5%               | 2     | [link](https://huggingface.co/Brendan/refpydst-5p-icdst-split-v2)           |
| `Brendan/refpydst-5p-icdst-split-v3`            | 5%               | 3     | [link](https://huggingface.co/Brendan/refpydst-5p-icdst-split-v3)           |

You can download **all of them** to the `REFPYDST_OUTPUTS_DIR` with:

```bash
python download_pretrained_retrievers.py
```

### Running Codex Experiments

#### Table 1: Few-shot Experiments

You can repeat our few-shot experiments on MultiWOZ 2.4 with:

```bash
# MultiWOZ 2.4
bash codex_expmt_folder_runner.sh runs/codex/mw21_1p_train/python
bash codex_expmt_folder_runner.sh runs/codex/mw21_5p_train/python
bash codex_expmt_folder_runner.sh runs/codex/mw21_10p_train/python
bash codex_expmt_folder_runner.sh runs/codex/mw21_100p_train/python
```

and evaluate on MultiWOZ 2.1:


```bash
# MultiWOZ 2.1 simulated from 2.4 result
bash mw21_simulation_expmt_folder_runner.sh runs/codex/mw21_1p_train/sim_mw21_python
bash mw21_simulation_expmt_folder_runner.sh runs/codex/mw21_5p_train/sim_mw21_python
bash mw21_simulation_expmt_folder_runner.sh runs/codex/mw21_10p_train/sim_mw21_python
bash mw21_simulation_expmt_folder_runner.sh runs/codex/mw21_100p_train/sim_mw21_python
```
#### Table 2 & 3: Zero-shot Experiments

You can repeat our zero-shot experiments on MultiWOZ 2.4 with:

```bash
# MultiWOZ 2.4
bash codex_expmt_folder_runner.sh runs/codex/zero_shot/python
```

and evaluate on MultiWOZ 2.1 with:

```bash
bash mw21_simulation_expmt_folder_runner.sh runs/codex/zero_shot/sim_mw21_python
```

#### Table 4: Ablations

Run all ablations in Table 4 (zero-shot and 5% few-shot) with:

This runs each in sequence by recurrsively walking the directories, which you can speed up by running separate commands
pointing at sub-directories. This will also include a random-retriever, as in the Appendix.

```bash
# MultiWOZ 2.4
bash codex_expmt_folder_runner.sh runs/table4
```

#### Table 5: Co-reference Analysis

To repeat experiments for Table 5, one can run:

```bash
bash codex_expmt_folder_runner.sh runs/table5
```

Once these runs complete, you can evaluate with [`eval_coref_turns.py`](./src/refpydst/eval_coref_turns.py).

Different from other scripts, this takes the **group name from wandb** as an argument for evaluating 
the co-reference performance of completed runs in that group. The script will verify the expected number of 
runs for the group, so make sure the runs you expect to be evaluated are marked in wandb with the tag `complete_run`
(should occur automatically on run finish), and mark any you wish to ignore with tag `outdated`.

An example call to the script:

```bash
python src/refpydst/eval_coref_turns.py "-runs-table5-5p-full"
```

will score the co-reference accuracy of the runs in `runs/table5/5p/full`.
