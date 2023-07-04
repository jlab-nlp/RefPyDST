import unittest

from refpydst.db.ontology import Ontology
from refpydst.normalization.data_ontology_normalizer import DataOntologyNormalizer


class DataOntologyNormalizerTests(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.ontology = Ontology.create_ontology()

    def test_simple_example(self):
        normalizer: DataOntologyNormalizer = DataOntologyNormalizer(
            ontology=self.ontology,
            counts_from_ontology_file="db/multiwoz/2.4/ontology.json"
        )
        self.assertEqual(
            normalizer.normalize({"hotel-name": "the acron guest houe"}),
            {"hotel-name": "acorn guest house"}
        )


if __name__ == '__main__':
    unittest.main()
