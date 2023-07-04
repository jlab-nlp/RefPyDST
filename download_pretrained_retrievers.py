import os

from huggingface_hub import HfApi, snapshot_download

REFPYDST_OUTPUTS_DIR: str = "REFPYDST_OUTPUTS_DIR"

if __name__ == '__main__':
    if REFPYDST_OUTPUTS_DIR not in os.environ:
        raise ValueError(f"Before downloading pre-trained models, set the {REFPYDST_OUTPUTS_DIR} to the absolute path "
                         f"where you want the retrievers to be stored. As a default from repo root, you can use: "
                         f"export {REFPYDST_OUTPUTS_DIR}=\"`pwd`/outputs\"")
    hf_api: HfApi = HfApi()

    # start with few-shot settings + each run
    few_shot_settings = ["mw21_1p_train", "mw21_5p_train", "mw21_10p_train"]
    run_versions = ["split_v1", "split_v2", "split_v3"]
    for data_size in few_shot_settings:
        for run_number in run_versions:
            download_path: str = os.path.join(os.environ[REFPYDST_OUTPUTS_DIR],
                                              f"runs/retriever/{data_size}/referred_states/{run_number}/")
            model_name = f"refpydst-{data_size.split('_')[1]}-referredstates-{run_number.replace('_', '-')}"
            snapshot_download(repo_id=f"Brendan/{model_name}", local_dir=download_path, local_dir_use_symlinks=False)

    # 100% retriever
    download_path: str = os.path.join(os.environ[REFPYDST_OUTPUTS_DIR],
                                      f"runs/retriever/mw21_100p_train/referred_states")
    snapshot_download(repo_id=f"Brendan/refpydst-100p-referredstates", local_dir=download_path,
                      local_dir_use_symlinks=False)

    # IC-DST 5% retrievers (for Table 4 ablation)
    for run_number in run_versions:
        download_path: str = os.path.join(os.environ[REFPYDST_OUTPUTS_DIR],
                                          f"runs/retriever/mw21_5p_train/ic_dst/{run_number}/")
        model_name = f"refpydst-5p-icdst-{run_number.replace('_', '-')}"
        snapshot_download(repo_id=f"Brendan/{model_name}", local_dir=download_path, local_dir_use_symlinks=False)
