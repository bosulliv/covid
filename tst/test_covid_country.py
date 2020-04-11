import unittest
import src.covid as covid


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
                                     filepath='./data/raw/',
                                     fixes=fix_data)
        self.uk.load()

    def test_smoke(self):
        """ Test unittest is setup """
        self.assertEqual(1, 1)

    def test_gamma_pdf_begin(self):
        """ Test the Gamma PDF function """
        y = covid.gamma_pdf(0, k=10, theta=1)
        self.assertAlmostEqual(0, y, places=1)

    def test_gamma_pdf_middle(self):
        """ Test the Gamma PDF function """
        y = covid.gamma_pdf(10, k=10, theta=1)
        self.assertAlmostEqual(0.125, y, places=2)

    def test_gamma_pdf_end(self):
        """ Test the Gamma PDF function """
        y = covid.gamma_pdf(20, k=10, theta=1)
        self.assertAlmostEqual(0, y, places=1)
        
    def test_gamma_pred_case_begin(self):
        """ Test the Gamma PDF function """
        y = covid.gamma_pred_case(0, duration=100, theta=1, peak=160000, spread=20)
        self.assertAlmostEqual(0, y/160000, places=1)

    def test_gamma_pred_case_middle(self):
        """ Test the Gamma PDF function """
        y = covid.gamma_pred_case(50, duration=100, theta=1, peak=160000, spread=20)
        self.assertAlmostEqual(0.5, y/160000, places=1)

    def test_gamma_pred_case_end(self):
        """ Test the Gamma PDF function """
        y = covid.gamma_pred_case(100, duration=100, theta=1, peak=160000, spread=20)
        self.assertAlmostEqual(1.0, y/160000, places=1)

if __name__ == '__main__':
    unittest.main(warnings='ignore')
