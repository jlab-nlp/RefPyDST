import json
import os
from glob import glob
from typing import Any, Optional, List

import wandb
from wandb.apis.public import Run

from refpydst.data_types import Turn
from refpydst.utils.general import REFPYDST_OUTPUTS_DIR, WANDB_PROJECT, WANDB_ENTITY


def output_dir_to_run_or_artifact_name(output_dir: str) -> str:
    parent_dir = REFPYDST_OUTPUTS_DIR in os.environ and os.environ[REFPYDST_OUTPUTS_DIR] or "outputs/"
    return output_dir.replace("../expts/", "").replace(parent_dir, "").replace('/', '-')\
        .replace("-data-users-bking2-rec_dst-expts-runs", "")


def write_json_artifact(jsonable_object: Any, output_dir: str, file_name: str, artifact_name: str,
                        artifact_type: str = "run_log", run: Optional[Run] = None,
                        project: str = "refpydst") -> wandb.Artifact:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path: str = os.path.join(output_dir, file_name)
    with open(file_path, 'w') as f:
        json.dump(jsonable_object, f)
    artifact: wandb.Artifact = wandb.Artifact(artifact_name, type=artifact_type, metadata={"output_dir": output_dir})
    artifact.add_dir(output_dir)
    if run is not None:
        run.log_artifact(artifact)
    else:
        artifact.save(project=project)
    return artifact


def get_json_artifact_by_file_name(expected_file_path: str) -> Any:
    output_dir, file_name = expected_file_path.rsplit("/", maxsplit=1)
    artifact_name: str = output_dir_to_run_or_artifact_name(output_dir)
    try:
        return read_json_artifact(artifact_name, file_name)
    except BaseException as e:
        artifact_name = "-data-users-bking2-rec_dst-expts-runs" + artifact_name
        return read_json_artifact(artifact_name, file_name)


def read_json_artifact(artifact_name: str, file_path: str,
                       alias: str = 'latest', project: str = "refpydst",
                       entity: str = "kingb12") -> Any:
    entity = os.environ.get(WANDB_ENTITY, entity)
    project = os.environ.get(WANDB_PROJECT, project)
    api = wandb.Api()
    artifact: wandb.Artifact = api.artifact(f'{entity}/{project}/{artifact_name}:{alias}')
    download_path: str = artifact.download()
    with open(os.path.join(download_path, file_path), "r") as f:
        return json.load(f)


def upload_all_artifacts(path: str, print_only: bool = False) -> None:
    hits = glob(f"{path}/**/running_log.json", recursive=True)
    for file_path in hits:
        if print_only:
            print(file_path)
            continue
        output_dir, file_name = file_path.rsplit("/", maxsplit=1)
        with open(file_path, 'r') as f:
            data = json.load(f)
        write_json_artifact(data, output_dir=output_dir, file_name=file_name,
                            artifact_name=output_dir_to_run_or_artifact_name(output_dir))


def read_run_artifact_logs(run_id: str, entity: str = "kingb12", project: str = "refpydst") -> Optional[List[Turn]]:
    """
    Get the running log associated with a wandb run id, if present
    :param run_id: run id (in url)
    :return: logs if an artifact of logs was added frmo the run
    """
    entity = os.environ.get(WANDB_ENTITY, entity)
    project = os.environ.get(WANDB_PROJECT, project)
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    for f in run.logged_artifacts():
        if f.type == 'run_output' or f.type == 'running_log':
            return read_json_artifact(f.name.split(':')[0], file_path="running_log.json", alias=f.version)


def get_running_logs_by_group(group_id: str, tags_in: List[str] = None, tags_not_in: List[str] = None) -> List[List[Turn]]:
    """
    Return the running logs associated with a group from wandb, subject to tag filters.

    :param group_id: group to get runs from
    :param tags_in: only get runs from the group if tagged with one of these tags. Defaults to ["complete_run"]
    :param tags_not_in: ignore any run tagged with one of these tags. Defaults to ["outdated"]
    :return: running logs from matching runs
    """
    tags_in = tags_in or ["complete_run"]
    tags_not_in = tags_not_in or ["outdated"]
    api = wandb.Api()
    entity = os.environ.get(WANDB_ENTITY, "kingb12")
    project = os.environ.get(WANDB_PROJECT, "refpydst")
    runs = api.runs(path=f"{entity}/{project}",
                    filters={"group": group_id, "tags": {"$in": tags_in, "$nin": tags_not_in}})
    result: List[List[Turn]] = []
    for run in runs:
        logs: List[Turn] = read_run_artifact_logs(run.id)
        result.append(logs)
    return result


if __name__ == '__main__':
    logs = read_run_artifact_logs("rotjhnz8")
    print(logs)