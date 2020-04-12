from math import gamma
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
from .utils_gamma import find_best_gamma_param, gamma_pred_case
from .utils_sigmoid import find_best_parameters, sig_pred_case
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
sns.set_style('whitegrid')



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
        self.best_peak = np.nan
        self.best_duration = np.nan
        self.best_theta = np.nan
        self.curve = 'gamma'
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
        self.curve = meta_df.loc[self.country, 'curve']
        
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
            curve='gamma',
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
        if curve:
            self.curve = curve
        if curve == 'sigmoid':
            values = find_best_parameters(self.country_df['Cases'],
                                          start_str=self.start_str,
                                          peak_guess=self.peak_guess,
                                          duration_guess=self.duration_guess,
                                          strategy='rmse',
                                          spread=16)
            self.best_peak, self.best_duration, self.best_score = values
        elif curve == 'gamma':
            values = find_best_gamma_param(self.country_df['Cases'],
                                           start_str=self.start_str,
                                           peak_guess=self.peak_guess,
                                           duration_guess=self.duration_guess,
                                           strategy='rmse',
                                           spread=20)
            self.best_peak, self.best_duration, self.best_theta, self.best_score = values

    def predict(self):
        """ Make predictions on model """
        pred_lst = []
        duration = self.best_duration
        peak = self.best_peak
        start_str = self.start_str
        theta = self.best_theta
        if self.curve == 'sigmoid':
            for i in range(0, duration):
                pred_lst.append(sig_pred_case(i, duration, peak))
            pred_df = pd.DataFrame(pred_lst,
                                   index=pd.date_range(start_str,
                                                       periods=duration))
        elif self.curve == 'gamma':
            for i in range(0, duration):
                pred_lst.append(gamma_pred_case(i, theta, duration, peak))
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
        ax.plot(df['Actual'], marker='o')
        ax.plot(df['Prediction'], linestyle='-.')
        #sns.lineplot(data=df,
        #             ax=ax,
        #             marker='o')
        title_str = f'{country}\nStart: {start_str}'
        title_str += f'\nDuration: {best_duration:.0f}'
        title_str += f'\nPeak: {best_peak:.0f}\nR-squared: {r2:.3f}'
        if self.curve == 'gamma':
            title_str += f'\nTheta: {self.best_theta:.2f}'
        plt.title(title_str)
        plt.tight_layout()
        plt.show()

    def _plot_daily_cases(self, start, stop):
        country = self.country
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        df = self.pred_df[start:stop].diff()
        ax.plot(df['Actual'], marker='o')
        ax.plot(df['Prediction'], linestyle='-.')
        plt.title(f'{country}\nEstimated Daily new cases')
        plt.tight_layout()
        plt.show()

    def save(self, filepath=None):
        """ Save the prediction file """
        if not filepath:
            filepath = self.path
        suffix = '_df.csv'
        filename = filepath + self.iata_2 + suffix
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
