from math import gamma
import numpy as np
import pandas as pd

def rmse(series):
    """ Return RMSE of an array of residuals """
    score = np.sum(series**2)**0.5
    return score

def loss_function(series, strategy='rmse'):
    """
    Given an array of residuals, return a loss score.

    Parameters
    ----------
    series : pd.Series,
        A Pandas series of residuals.
    """
    series = series.dropna()
    if series.shape[0] == 0:
        score = np.nan
    else:
        score = rmse(series)
    return score

def gamma_pdf(x, mu=50, theta=1):
    """ A function with the same shape as the Gamma PDF. It returns y
    for a given x, and the shape parameters described.
    
    Parameters
    ----------
    x : int
        The x position to return y=gamma_pdf(y)
        
    mu : float
        The mean = k * theta. In the first spread, this would be
        roughly half the total duration. And a little sooner with skew.
        
    theta : float
        The standard deviation
    """
    k = mu / theta
    nearly_zero = 0.1**100
    if x == 0:
        x = nearly_zero
    alpha = k
    beta = 1/theta
    numer = beta**alpha
    numer *= x**(alpha-1)
    numer *= np.exp(-1*beta*x)
    denom = gamma(alpha)
    return numer/denom

def gamma_pred_case(i, theta=0.75, duration=90, peak=160000, spread=20):
    """ Return the predicted total cases for day 'i' using a gamma
    distributon function with the given parameters. """
    x_final = (i*spread)/duration
    case_total = 0
    scale = peak*spread/duration
    for x in np.linspace(0, x_final, i):
        mu = spread/2
        #k = spread/(2*theta)
        y = gamma_pdf(x=x, mu=mu, theta=theta)
        case_total += y*scale
    return case_total

def find_best_gamma_param(df,
                          start_str,
                          peak_guess,
                          duration_guess,
                          spread=20,
                          strategy='rmse'):
    score_df = pd.DataFrame({'peak': [],
                             'duration': [],
                             'theta': [],
                             'score': []})
    score_df.set_index(['peak', 'duration', 'theta'], inplace=True)
    peak_grid = np.linspace(int(0.75*peak_guess),
                            int(1.25*peak_guess),
                            10)
    duration_grid = range(duration_guess-2,
                          duration_guess+5,
                          1)
    theta_grid = np.linspace(0.5, 2.0, 16)
    for peak in peak_grid:
        for duration in duration_grid:
            for theta in theta_grid:
                gamma_case_lst = []
                for i in range(0, duration):
                    value = gamma_pred_case(i,
                                            theta=theta,
                                            duration=duration,
                                            peak=peak,
                                            spread=spread)
                    gamma_case_lst.append(value)

                current_df = pd.DataFrame(gamma_case_lst,
                                          index=pd.date_range(start_str,
                                                              periods=duration))

                conc_df = pd.concat([df[start_str:], current_df], axis=1)
                conc_df.columns = ['Actual', 'Prediction']
                conc_df['residual'] = conc_df['Actual'] - conc_df['Prediction']
                score = loss_function(conc_df['residual'])
                score_df.loc[(peak, duration, theta), 'score'] = score

    best_peak, best_duration, best_theta = score_df['score'].idxmin()
    best_score = score_df.loc[(best_peak, best_duration, best_theta), 'score']
    return int(best_peak), int(best_duration), best_theta, best_score
