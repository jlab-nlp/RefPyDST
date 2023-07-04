import unittest

from refpydst.db.ontology import Ontology


class OntologyTests(unittest.TestCase):
    def test_simple_get_canonical(self):
        ontology: Ontology = Ontology.create_ontology()
        # Basic usage: convert surface forms to canonical forms. Will return None if the surface form cannot be mapped
        self.assertEqual(ontology.get_canonical("hotel-name", "the acron guest house"), "acorn guest house")
        self.assertEqual(ontology.get_canonical("hotel-name", "this is not a real hotel"), None)


if __name__ == '__main__':
    unittest.main()
