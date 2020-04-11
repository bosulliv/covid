import unittest
from src.covid import CovidCountry


class CovidCountryTest(unittest.TestCase):
    """
    From project root:

        $ python -m unittest discover

    And that will run these tests.
    """
    def setUp(self):
        # Load and fix UK data
        fix_data = {'2020-03-12': 590,
                    '2020-03-15': 1391,
                    '2020-03-19': 3269}

        self.uk = CovidCountry(country='United Kingdom',
                               filepath='./data/raw/',
                               fixes=fix_data)
        self.uk.load()

    def test_smoke(self):
        """ Test unittest is setup """
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
