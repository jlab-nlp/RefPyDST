from typing import Dict, Any

from data_types import MultiWOZDict
from normalization.abstract_normalizer import AbstractNormalizer
from refpydst.utils.ic_dst_typo_fix import typo_fix


class ICDSTNormalizer(AbstractNormalizer):
    """
    A class encoding the baseline 'normalizer' from:

    @article{hu2022context,
      title={In-Context Learning for Few-Shot Dialogue State Tracking},
      author={Hu, Yushi and Lee, Chia-Hsuan and Xie, Tianbao and Yu, Tao and Smith, Noah A and Ostendorf, Mari},
      journal={arXiv preprint arXiv:2203.08568},
      year={2022}
    }
    """
    ontology: Dict[str, Any]
    version: str

    def __init__(self, ontology_json: Dict[str, Any], mwz_version: str):
        self.ontology = ontology_json
        self.version = mwz_version

    def normalize(self, raw_parse: MultiWOZDict) -> MultiWOZDict:
        return typo_fix(raw_parse, ontology=self.ontology, version=self.version)
