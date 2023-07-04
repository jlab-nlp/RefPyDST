import logging
import os
import pprint
import sys
from typing import List, Dict, Any

import wandb
from refpydst.data_types import Turn, CompletionParser, SlotName
from refpydst.prompting import PromptGenerator, get_completion_parser
from tqdm import tqdm

import refpydst.artifacts as artifacts
from refpydst.db.ontology import Ontology
from refpydst.evaluate_run_log import evaluate_logs, evaluate_on_domains
from refpydst.normalization.abstract_normalizer import AbstractNormalizer
from refpydst.normalization.data_ontology_normalizer import DataOntologyNormalizer
from refpydst.normalization.ic_dst_normalizer import ICDSTNormalizer
from refpydst.resources import read_json_resource
from refpydst.utils.dialogue_state import equal_dialogue_utterances
from refpydst.utils.dialogue_state import group_by_dial_id_and_turn, update_dialogue_state
from refpydst.utils.general import read_json, get_output_dir_full_path, read_json_from_data_dir, WANDB_ENTITY, \
    WANDB_PROJECT
from refpydst.utils.state_recorder import PreviousStateRecorder


def _get_normalizer(normalizer_type: str = None, train_set: List[Turn] = None):
    # we package these with the repo as json resource files
    ontology_json_resource_path: str = "db/multiwoz/2.1/ontology.json"
    if normalizer_type == "ic_dst":
        # read the json file in before passing to normalizer
        ontology_json: Dict[SlotName, List[str]] = read_json_resource(ontology_json_resource_path)
        return ICDSTNormalizer(ontology_json=ontology_json, mwz_version="2.1")
    elif normalizer_type == "data_ontology":
        if not train_set:
            train_set = []
        ontology = Ontology.create_ontology()
        # this normalizer will read in from path
        return DataOntologyNormalizer(ontology=ontology, supervised_set=train_set,
                                      counts_from_ontology_file=ontology_json_resource_path)
    else:
        raise ValueError(f"invalid normalizer type: {normalizer_type}")


class SimulatedCodexExperiment:
    logs: List[Turn]
    mw21_test_set: List[Turn]
    mw21_train_set: List[Turn]
    train_by_dial_id_and_turn: Dict[str, List[Turn]]

    normalizer: AbstractNormalizer
    prompt_generator: PromptGenerator
    prompt_format: str
    prediction_recorder: PreviousStateRecorder
    completion_parser: CompletionParser

    def __init__(self, artifact_id: str = None, log_file: str = None, artifact_alias: str = "latest",
                 train_fn: str = None,
                 test_fn: str = None,
                 normalizer_type: str = "data_ontology",
                 prompt_format: str = None,
                 verify_pred: bool = False, **kwargs) -> None:
        super().__init__()
        if artifact_id:
            self.logs = artifacts.read_json_artifact(artifact_name=artifact_id, file_path="running_log.json",
                                                     alias=artifact_alias)
        else:
            self.logs = read_json(log_file)

        self.mw21_test_set = read_json_from_data_dir(test_fn)
        self.mw21_train_set = read_json_from_data_dir(train_fn)
        self.train_by_dial_id_and_turn = group_by_dial_id_and_turn(self.mw21_train_set)
        self.normalizer = _get_normalizer(normalizer_type, self.mw21_train_set)
        self.prompt_format = prompt_format
        self.prompt_generator = PromptGenerator()
        self.prediction_recorder = PreviousStateRecorder()
        self.completion_parser = get_completion_parser(self.prompt_format)
        self.verify_pred = verify_pred

    def run(self):
        for log, turn in tqdm(zip(self.logs, self.mw21_test_set),
                              desc="simulating and verifying MultiWOZ 2.1 experiment from 2.4 result"):
            # These ensure we are in fact testing a fair log for mw21
            assert log['ID'] == turn['ID']
            assert log['turn_id'] == turn['turn_id']
            assert equal_dialogue_utterances(log, turn)
            predicted_context = self.prediction_recorder.retrieve_previous_turn_state(turn)
            if 'examples' in log and 'formatting' not in log['examples'][0][0]:
                examples: List[Turn] = [self.train_by_dial_id_and_turn[dial_id][turn_num]
                                        for dial_id, turn_num in log['examples']]
                prompt_text = self.prompt_generator.get_prompt(turn, examples=examples, given_context=predicted_context,
                                                               n_examples=len(examples),
                                                               prompt_format=self.prompt_format)
                # due non-determinism in ordering of printed context slot value pairs, have to remove states from
                # this check
                prompt_text = "\n".join([line for line in prompt_text.splitlines() if not "[context]" in line])
                orig_prompt_text = "\n".join([line for line in log['prompt'].splitlines() if not "[context]" in line])
                assert prompt_text == orig_prompt_text
            predicted_update = self.completion_parser(log['completion'], predicted_context)
            predicted_update = self.normalizer.normalize(predicted_update)
            all_slot_values = update_dialogue_state(predicted_context, predicted_update)
            # some slots may contain multiple values
            all_slot_values = {k: v.split('|')[0] for k, v in all_slot_values.items()}
            self.prediction_recorder.add_state(turn, all_slot_values)
            if not self.normalizer.normalize(log['pred']) == all_slot_values:
                if self.verify_pred:
                    raise ValueError(f"expected equal predictions: {pprint.pformat(log['pred'])}, "
                                     f"{pprint.pformat(all_slot_values)}")
                else:
                    logging.warning(f"found non-equal predictions: {pprint.pformat(log['pred'])}, "
                                    f"{pprint.pformat(all_slot_values)}")
            log['pred'] = all_slot_values
        stats = evaluate_logs(self.logs, test_set=self.mw21_test_set)
        wandb.log(stats)
        # add by-domain evaluation
        by_domain_stats = evaluate_on_domains(self.logs, self.mw21_test_set)
        flattened_domain_stats: Dict[str, Any] = {}
        for domain, domain_scores in by_domain_stats.items():
            for metric, value in domain_scores.items():
                flattened_domain_stats[f"{domain}-{metric}"] = value
        wandb.log(flattened_domain_stats)


def main(artifact_id: str = None, artifact_alias: str = "latest", train_fn: str = None, test_fn: str = None,
         normalizer_type: str = "data_ontology", prompt_format: str = None, **kwargs) -> None:
    experiment: SimulatedCodexExperiment = SimulatedCodexExperiment(
        artifact_id=artifact_id,
        artifact_alias=artifact_alias,
        train_fn=train_fn,
        test_fn=test_fn,
        normalizer_type=normalizer_type,
        prompt_format=prompt_format,
        **kwargs
    )
    experiment.run()


if __name__ == "__main__":
    if os.path.exists(sys.argv[1]):
        run_file: str = sys.argv[1]
        # arguments are input from a configuration file if the first argument to the program is a valid file
        args = read_json(run_file)
        if 'output_dir' not in args:
            args['output_dir'] = get_output_dir_full_path(run_file.replace('.json', ''))
        if not 'run_name' in args:
            args['run_name'] = artifacts.output_dir_to_run_or_artifact_name(args['output_dir'])
    default_run_name: str = artifacts.output_dir_to_run_or_artifact_name(args['output_dir'])
    default_run_group: str = default_run_name.rsplit('-', maxsplit=1)[0]
    wandb_entity: str = os.environ.get(WANDB_ENTITY, "kingb12")
    wandb_project: str = os.environ.get(WANDB_PROJECT, "refpydst")
    run = wandb.init(config=args, project=wandb_project, entity=wandb_entity,
                     name=args.get("run_name", default_run_name), notes=args.get("run_notes", None),
                     group=args.get("run_group", default_run_group))
    main(**args)
