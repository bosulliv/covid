import numpy as np
import pandas as pd


def loss_function(series, strategy='rmse'):
    """
    Given a Series of residuals, return a loss score.

    Parameters
    ----------
    series : pd.Series,
        A Pandas series of residuals.

    strategy : str, default='rmse'
        How to measure loss. RMSE, MSE, and custom functions.
    """
    series = series.dropna()
    if series.shape[0] == 0:
        score = np.nan
    #elif abs(series.sum()) < 0.001:
    #    print(series)
    #    raise Exception('Zero residual sum not possible')
    elif strategy == 'mse':
        score = np.sum(series**2)
    elif strategy == 'rmse':
        score = np.sum(series**2)**0.5
    elif strategy == 'best_fit_last':
        # rmse - but with a final multiple which
        # strongly biases more recent residuals
        half = int(round(len(series)/2.0))
        blank = np.zeros(half)
        mult = np.linspace(1, 2, int(len(series)-half))**2
        mult = np.concatenate([blank, mult])
        score = series**2
        score = np.multiply(mult, score)
        score = np.sum(score)**0.5
    return score

def sig_pred_case(i, duration=70, peak=80000, spread=16):
    """ Return the value of the sigmoid function at the point i.

    Parameters
    ----------
    i : int
        This is the day since the start of an outbreak. Return the
        case count at this index.

    duration: int
        This is the tail to tail duration of the outbreak.

    peak: int
        This is the maximum value of sigmoid function.

    spreak: int
        This governs the limits of x inside np.exp(x) - x will vary
        between -spread/2 and +spread/2 e.g. for spread = 16
        the first day of outbreak will take value -8 and the last day
        of outbreak will be +8 inside np.exp(x)
    """
    numer = peak
    # exp(0) happens at 50% time of infection
    index = i - duration/2
    # np.exp(y) - we want y to be between +/- spread/2
    lam = spread/duration
    denom = (1+np.exp(-1*lam*index))
    sig_num = numer/denom
    return sig_num

def find_best_parameters(df,
                         start_str,
                         peak_guess,
                         duration_guess,
                         spread=16,
                         strategy='rmse'):
    score_df = pd.DataFrame({'peak': [],
                             'duration': [],
                             'score': []})
    score_df.set_index(['peak', 'duration'], inplace=True)
    peak_grid = range(int(0.5*peak_guess),
                      int(1.5*peak_guess),
                      int(0.025*peak_guess))
    duration_grid = range(duration_guess-4, duration_guess+3)
    for peak in peak_grid:
        for duration in duration_grid:
            sig_case_lst = []
            for i in range(0, duration):
                value = sig_pred_case(i, duration, peak, spread=spread)
                sig_case_lst.append(value)

            current_df = pd.DataFrame(sig_case_lst,
                                      index=pd.date_range(start_str,
                                                          periods=duration))

            conc_df = pd.concat([df[start_str:], current_df], axis=1)
            conc_df.columns = ['Actual', 'Prediction']
            conc_df['residual'] = conc_df['Actual'] - conc_df['Prediction']
            score = loss_function(conc_df['residual'], strategy=strategy)
            score_df.loc[(peak, duration), 'score'] = score

    best_peak, best_duration = score_df['score'].idxmin()
    best_score = score_df.loc[(best_peak, best_duration), 'score']
    return int(best_peak), int(best_duration), best_score
