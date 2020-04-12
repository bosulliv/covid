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
    """
    def test_smoke(self):
        """ Test unittest is setup """
        self.assertEqual(1, 1)

    def test_gamma_pdf_begin(self):
        """ Test the Gamma PDF function """
        y = covid.gamma_pdf(0, mu=50, theta=1)
        self.assertAlmostEqual(0, y, places=1)

    def test_gamma_pdf_middle(self):
        """ Test the Gamma PDF function. The midpoint is a little
            before the mean as the function is skewed. """
        y = covid.gamma_pdf(47, mu=50, theta=1)
        self.assertAlmostEqual(0.05, y, places=2)

    def test_gamma_pdf_end(self):
        """ Test the Gamma PDF function """
        y = covid.gamma_pdf(100, mu=50, theta=1)
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

    def test_best_gamma_params(self):
        """ Test fit against a known curve """
        peak = 160000
        duration = 100
        spread=20
        theta=1
        start_str='2020-02-23'
        
        # Build perfect data
        y = []
        for day in range(100):
            val = covid.gamma_pred_case(day, duration=duration,
                                        theta=theta, peak=peak,
                                        spread=spread)
            y.append(val)
        dt_idx = pd.date_range(start_str, freq='1D', periods=duration)
        df = pd.DataFrame({'Actual': y}, index=dt_idx)
        
        # Prove we can fit and find the same parameters
        values = covid.find_best_gamma_param(df=df,
                                             start_str=start_str,
                                             spread=spread,
                                             peak_guess=peak,
                                             duration_guess=duration)
        # The way I grid search peak means it is very unlikely to be equal
        # It just needs to be close
        self.assertLess(abs(peak-values[0]), peak*0.05)
        self.assertLess(abs(duration-values[1]), duration*0.05)
        self.assertLess(abs(theta-values[2]), theta*0.1)

if __name__ == '__main__':
    unittest.main(warnings='ignore')
