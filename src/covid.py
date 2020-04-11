from math import gamma
from urllib.request import urlretrieve
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_context('talk')
sns.set_style('whitegrid')


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
    elif abs(series.sum()) < 0.001:
        print(series)
        raise Exception('Zero residual sum not possible')
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

def gamma_pdf(x, k=10, theta=0.75):
    """ A function with the same shape as the Gamma PDF. It returns y
    for a given x, and the shape parameters described.
    
    Parameters
    ----------
    x : int
        The x position to return y=gamma_pdf(y)
        
    k : float
        The mean
        
    theta : float
        The standard deviation
    """
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
    for x in np.linspace(0, x_final, i):
        k = spread/(2*theta)
        scale = peak*spread/duration
        y = gamma_pdf(x=x, k=k, theta=theta)
        case_total += y*scale
    return case_total

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

def find_best_gamma_param(df,
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
    duration_grid = range(duration_guess-4,
                          duration_guess+3,
                          1)
    theta_grid = np.linrange(0.25, 1.75, 20)
    for peak in peak_grid:
        for duration in duration_grid:
            for theta in theta_grid:
                gamma_case_lst = []
                for i in range(0, duration):
                    value = gamma_pred_case(i, duration=duration,
                                            theta=theta, peak=peak,
                                            spread=spread)
                    gamma_case_lst.append(value)

                current_df = pd.DataFrame(gamma_case_lst,
                                          index=pd.date_range(start_str,
                                                              periods=duration))

                conc_df = pd.concat([df[start_str:], current_df], axis=1)
                conc_df.columns = ['Actual', 'Prediction']
                conc_df['residual'] = conc_df['Actual'] - conc_df['Prediction']
                score = loss_function(conc_df['residual'], strategy=strategy)
                score_df.loc[(peak, duration, theta), 'score'] = score

    best_peak, best_duration, best_theta = score_df['score'].idxmin()
    best_score = score_df.loc[(best_peak, best_duration, best_theta), 'score']
    return int(best_peak), int(best_duration), best_theta, best_score

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
    duration_grid = range(duration_guess-4,
                          duration_guess+3,
                          1)
    for peak in peak_grid:
        for duration in duration_grid:
            sig_case_lst = []
            for i in range(0, duration):
                value = sig_pred_case(i, duration, peak, spread=spread)
                value = gamma_pred_case(i, duration, peak, spread=spread)
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


class CovidCountry():
    """
    A self contained class to download covid data and fit and display
    predictions.

    Parameters
    ----------
    country : str, default='United Kingdom'
        The name of the country, or region to filter from the raw data.

    province : str, default=False
        Many countries have multiple regions, which are very far apart.
        In this case, you can just explore one region. For instance,
        the UK has spread territories, but for virus transmission it
        doesn't make sense to group those together. It makes more sense
        just to analyse areas that are continious.

    Attributes
    ----------
    tidy_df : pd.DataFrame
        The downloaded, processed and corrected data.
    """

    def __init__(self, country='United Kingdom', verbose=False,
                 filepath='./data/raw/', fixes={}):
        """ Init - see class doc for details """
        self.verbose = verbose
        self.country = country
        self.province = False
        self.population = np.nan
        self.iata_2 = ''
        self.url = 'https://raw.githubusercontent.com/'
        self.url += 'CSSEGISandData/COVID-19/master/'
        self.url += 'csse_covid_19_data/csse_covid_19_time_series/'
        self.url += 'time_series_covid19_confirmed_global.csv'
        self.path = filepath
        self.file = 'ts_covid.csv'
        self.fixes = fixes
        self.raw_df = pd.DataFrame()
        self.tidy_df = pd.DataFrame()
        self.country_df = pd.DataFrame()
        self.pred_df = pd.DataFrame()
        self.best_peak = 150000
        self.best_duration = 90
        self.r2 = np.nan
        self.start_str = ''

    def load(self, today=None):
        """ Download the latest data """
        self._get_meta_data()
        self._download()
        self._engineer()
        self._filter()
        self._correct()
        if today:
            self._add_today(today)

    def _get_meta_data(self):
        """ Automatically load the given countries meta_data """
        filename = self.path + 'country_meta.csv'
        meta_df = pd.read_csv(filename, index_col='country')
        try:
            meta_df.loc[self.country,:]
        except:
            raise ValueError(f'No data for country {self.country}')
        self.start_str = meta_df.loc[self.country, 'start_str']
        self.province = meta_df.loc[self.country, 'province']
        self.duration_guess = meta_df.loc[self.country, 'duration_guess']
        self.peak_guess = meta_df.loc[self.country, 'peak_guess']
        self.population = meta_df.loc[self.country, 'start_str']
        self.iata_2 = meta_df.loc[self.country, 'iata_2']
        
    def _download(self):
        """ Download latest covid case data. """
        urlretrieve(self.url, self.path+self.file)
        # sanity check data size - changes need to be checked
        df = pd.read_csv(self.path+self.file)
        col_count = pd.Timestamp.now().now() - pd.to_datetime('2020-01-22')
        col_count /= pd.Timedelta('1D')
        col_count += 4
        col_count = np.floor(col_count)

        if self.verbose:
            print(df.shape)

        if df.shape[1] != col_count:
            raise ValueError('More columns than expected')
        if df.shape[0] != 264:
            raise ValueError(f'More rows than expected. {df.shape}')
        self.raw_df = df

    def _engineer(self):
        """ Transform raw data into model shape.
        One row per country, per day """
        # Set blank province to the country/region name
        df = self.raw_df
        idx = df['Province/State'].isna()
        df.loc[idx, 'Province/State'] = df.loc[idx, 'Country/Region']

        id_vars = ['Province/State', 'Lat', 'Long', 'Country/Region']
        tidy_df = df.melt(id_vars=id_vars,
                          value_name='Cases',
                          var_name='Date').reset_index()
        tidy_df['Date'] = pd.to_datetime(tidy_df['Date'], format='%m/%d/%y')
        tidy_df = tidy_df.set_index('Date').sort_index()
        if self.verbose:
            print(tidy_df['Country/Region'].value_counts().head(10))
        self.tidy_df = tidy_df

    def _filter(self):
        """ Create a country data frame """
        if self.province:
            idx = self.tidy_df['Province/State'] == self.country
        else:
            idx = self.tidy_df['Country/Region'] == self.country
        self.country_df = self.tidy_df[idx].drop('index', axis=1)

    def _correct(self):
        """  Fix errors in data. """
        for date, value in self.fixes.items():
            self.country_df.loc[date, 'Cases'] = value

    def _add_today(self, cases):
        """ Add todays number to yesterdays data set """
        today_dt = pd.to_datetime(pd.Timestamp.now().date())

        if self.country_df[today_dt:].shape[0] == 1:
            # set the value
            idx = self.country_df.index == today_dt
            if idx.sum() == 0:
                print('Broken')
            self.country_df.loc[idx, 'Cases'] = cases
        elif self.country_df[today_dt:].shape[0] == 0:
            today = self.country_df.loc[today_dt - pd.Timedelta('1D'):].copy()
            today.index = [pd.Timestamp(today_dt)]
            today['Cases'] = cases
            self.country_df = pd.concat([self.country_df, today], axis=0)
        else:
            raise ValueError('Too many matching entries')

    def fit(self,
            start_str=None,
            duration_guess=None,
            peak_guess=None):
        """
        Fit model to the actual data so far. It does this by fitting
        a sigmoid curve to the total cases using grid search around
        the guess parameters provided.

        Parameters
        ----------
        start_date : string, e.g. '2020-02-01'
            The date day on day infection growth begins.

        guess_duration : int, e.g. 90
            Guess of tail to tail duration.

        guess_peak : int, e.g. 150000
            Guess of the total cases at the end of the outbreak.
        """
        if start_str:
            self.start_str = start_str
        if duration_guess:
            self.duration_guess = duration_guess
        if peak_guess:
            self.peak_guess = peak_guess
        values = find_best_parameters(self.country_df['Cases'],
                                      start_str=self.start_str,
                                      peak_guess=self.peak_guess,
                                      duration_guess=self.duration_guess,
                                      strategy='rmse',
                                      spread=16)
        self.best_peak, self.best_duration, self.best_score = values

    def predict(self):
        """ Make predictions on model """
        pred_lst = []
        duration = self.best_duration
        peak = self.best_peak
        start_str = self.start_str
        for i in range(0, duration):
            pred_lst.append(sig_pred_case(i, duration, peak))

        pred_df = pd.DataFrame(pred_lst,
                               index=pd.date_range(start_str,
                                                   periods=duration))

        actual_df = self.country_df.loc[start_str:, 'Cases']
        conc_df = pd.concat([actual_df, pred_df], axis=1)
        conc_df.columns = ['Actual', 'Prediction']
        self.pred_df = conc_df
        self._calc_r2()
        return self.pred_df

    def _calc_r2(self):
        from sklearn.metrics import r2_score
        df = self.pred_df.dropna()
        y = df['Actual']
        y_pred = df['Prediction']
        score = r2_score(y, y_pred)
        self.r2 = score

    def display(self, **kwargs):
        """ Show the model """
        offset = kwargs.get('offset', round(self.best_duration/8))
        start = pd.Timestamp.now().date() - pd.Timedelta(f'{offset}D')
        stop = pd.Timestamp.now().date() + pd.Timedelta(f'{offset}D')
        self._plot_total_cases(start, stop)
        self._plot_daily_cases(start, stop)

    def _plot_total_cases(self, start, stop):
        country = self.country
        start_str = self.start_str
        best_duration = self.best_duration
        best_peak = self.best_peak
        r2 = self.r2
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        df = self.pred_df[start:stop]
        sns.lineplot(data=df,
                     ax=ax,
                     marker='o')
        title_str = f'{country}\nStart: {start_str}'
        title_str += f'\nDuration: {best_duration:.0f}'
        title_str += f'\nPeak: {best_peak:.0f}\nR-squared: {r2:.3f}'
        plt.title(title_str)
        plt.tight_layout()
        plt.show()

    def _plot_daily_cases(self, start, stop):
        country = self.country
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        df = self.pred_df[start:stop].diff()
        df.plot(kind='bar', ax=ax)
        plt.title(f'{country}\nEstimated Daily new cases')
        plt.tight_layout()
        plt.show()

    def save(self):
        """ Save the prediction file """
        suffix = '_df.csv'
        filename = self.path + self.iata_2 + suffix
        self.pred_df.to_csv(filename)


if __name__ == '__main__':
    """ Make predictions on UK data and show plot """
    fixes = {'2020-03-12': 590,
             '2020-03-15': 1391,
             '2020-03-19': 3269,
             }
    c = CovidCountry(country='United Kingdom',
                     fixes=fixes)
    c.load(today=80000)
    c.fit()
    c.predict()
    c.display()
