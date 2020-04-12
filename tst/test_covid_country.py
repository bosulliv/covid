import unittest
import src.covid as covid
import numpy as np
import pandas as pd


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

        self.uk = covid.CovidCountry(country='United Kingdom',
                                     filepath='./data/test/',
                                     fixes=fix_data)
        self.uk.load()


if __name__ == '__main__':
    unittest.main(warnings='ignore')
