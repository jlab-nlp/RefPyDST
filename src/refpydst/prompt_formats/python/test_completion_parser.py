import unittest

from refpydst.data_types import CompletionParser
from refpydst.prompting import get_completion_parser, PYTHON_PROMPT

from refpydst.db.ontology import Ontology
from refpydst.normalization.data_ontology_normalizer import DataOntologyNormalizer
from refpydst.prompt_formats.python.completion_parser import replace_state_references


class CompletionParserTests(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.normalizer = DataOntologyNormalizer(ontology=Ontology.create_ontology())
        self.normalizer._counts_from_json("normalization/test_data/pre_loaded_normalizer_counts.json")

    def test_expected_parse(self):
        parser = get_completion_parser(PYTHON_PROMPT)
        self._check_parse(
            parser,
            'taxi = agent.find_taxi(leave_from="gallery at 12 a high street", arrive_by_time="15:30")\n\n',
            {},
            {'taxi-departure': 'gallery at 12 a high street', 'taxi-arriveby': '15:30'}
        )
        self._check_parse(parser,
                          """restaurant = agent.find_restaurant(book_day="same day", book_people=6, 
                                 book_time="11:15") """,
                          {'restaurant-area': 'centre',
                           'restaurant-book people': '6',
                           'restaurant-book time': '11:15',
                           'restaurant-food': 'chinese',
                           'restaurant-pricerange': 'expensive',
                           'train-day': 'saturday',
                           'train-departure': 'cambridge',
                           'train-destination': 'stevenage',
                           'train-leaveat': '14:00'},
                          {'restaurant-area': 'centre',
                           'restaurant-book people': '6',
                           'restaurant-book time': '11:15',
                           'restaurant-food': 'chinese',
                           'restaurant-pricerange': 'expensive',
                           'train-day': 'saturday',
                           'train-departure': 'cambridge',
                           'train-destination': 'stevenage',
                           'train-leaveat': '14:00'}
                          )

        self._check_parse(
            parser,
            'hotel = agent.find_hotel(name="the acorn guest house")',
            {'restaurant-pricerange': 'expensive', 'restaurant-food': 'north american', 'restaurant-area': 'centre',
             'restaurant-name': 'gourmet burger kitchen', 'restaurant-book time': '11:45',
             'restaurant-book day': 'sunday', 'restaurant-book people': '6'},
            {'restaurant-pricerange': 'expensive', 'restaurant-food': 'north american', 'restaurant-area': 'centre',
             'restaurant-name': 'gourmet burger kitchen', 'restaurant-book time': '11:45',
             'restaurant-book day': 'sunday', 'restaurant-book people': '6', 'hotel-name': 'acorn guest house'}
        )
        self._check_parse(
            parser,
            'taxi = agent.find_taxi(depart_time="04:15", destination=agent.state.hotel.name,\n'
            'leave_from = agent.state.attraction.name)\n\n',
            {'attraction-name': 'whipple museum of the history of science', 'hotel-name': 'alexander bed and breakfast',
             'hotel-book stay': '2', 'hotel-book day': 'friday', 'hotel-book people': '4'},
            {'hotel-name': 'alexander bed and breakfast', 'hotel-book stay': '2', 'hotel-book day': 'friday',
             'hotel-book people': '4', 'attraction-name': 'whipple museum of the history of science',
             'taxi-destination': 'alexander bed and breakfast',
             'taxi-departure': 'whipple museum of the history of science', 'taxi-leaveat': '04:15'}
        )
        self._check_parse(
            parser,
            'restaurant.food_type = "modern eupoean"\n\n',
            {'restaurant-food': 'modern european', 'restaurant-area': 'centre', 'restaurant-pricerange': 'moderate'},
            {'restaurant-food': 'modern european', 'restaurant-area': 'centre', 'restaurant-pricerange': 'moderate'}
        )
        self._check_parse(
            parser,
            'train = agent.find_train(destination="stansted airport", leave_from="cambridge")',
            {'train-departure': 'cambridge', 'train-destination': 'stansted airport'},
            {'train-departure': 'cambridge', 'train-destination': 'stansted airport'}
        )

    def test_replace_state_references(self):
        statement = 'taxi = Taxi(leave_from=" park fen causeway", destination=state.restaurant.name)'
        self.assertEqual(replace_state_references(statement), 'taxi = Taxi(leave_from=" park fen causeway", destination=agent.state.restaurant.name)')
        # dont add agent unnecessarily
        statement = 'taxi = Taxi(leave_from=" park fen causeway", destination=agent.state.restaurant.name)'
        self.assertEqual(replace_state_references(statement), 'taxi = Taxi(leave_from=" park fen causeway", destination=agent.state.restaurant.name)')

    def _check_parse(self, parser: CompletionParser, completion, context, expected_parse):
        actual_parse = self.normalizer.normalize(parser(completion, context))
        self.assertEqual(actual_parse, expected_parse)


if __name__ == '__main__':
    unittest.main()
