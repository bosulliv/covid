from urllib.request import urlretrieve
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_context('talk')


class CovidCountry():
    """
    A self contained class to download covid data and fit
    and display predictions.

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

    def __init__(self, country='United Kingdom',
                 province=False, verbose=False):
        """ Init - see class doc for details """
        self.verbose = verbose
        self.country = country
        self.province = province
        self.url = 'https://raw.githubusercontent.com/'
        self.url += 'CSSEGISandData/COVID-19/master/'
        self.url += 'csse_covid_19_data/csse_covid_19_time_series/'
        self.url += 'time_series_covid19_confirmed_global.csv'
        self.file = './data/raw/ts_covid.csv'
        self.raw_df = pd.DataFrame()
        self.tidy_df = pd.DataFrame()
        self.country_df = pd.DataFrame()

    def load(self):
        """ Download the latest data """
        self._download()
        self._engineer()
        self._filter()
        self._correct()

    def _download(self):
        """ """
        urlretrieve(self.url, self.file)
        # sanity check data size - changes need to be checked
        df = pd.read_csv(self.file)
        col_count = pd.Timestamp.now().now() - pd.to_datetime('2020-01-22')
        col_count /= pd.Timedelta('1D')
        col_count += 4
        col_count = np.floor(col_count)

        if self.verbose:
            print(df.shape)
        # TODO: change to raise
        if df.shape[1] != col_count:
            raise ValueError('More columns than expected')
        if df.shape[0] != 263:
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
        self.country_df = self.tidy_df[idx]

    def _correct(self, fixes={}):
        """ Fix errors in data on certain days """
        pass

    def fit(self):
        """ Fit model """
        pass

    def predict(self):
        """ Make predictions on model """
        pass

    def display(self):
        """ Show the model """
        fig, ax = plt.subplots(1, 1)


if __name__ == '__main__':
    c = CovidCountry(country='United Kingdom', province=True, verbose=False)
    c.load()
    print(c.country_df)
