# Covid
## Overview
Explore the covid data. Fit simple models. Update as daily figures arrive, and reflect on the changes necessary to improve the model fit.

## Country Models
Individual countries are modeled and ploted here:

* [UK](notebooks/UK.ipynb)
* [US](notebooks/US.ipynb)
* [Italy](notebooks/Italy.ipynb)

[This notebook](notebooks/covid.ipynb) is the original notebook, it's a bit messy, but has more countries at present. It is soon to be depreciated as the code has been refactored and moved into a module.


## Project Structure

<pre>
    /data      - virus time series, country metadata, and model paramater values
    /notebooks - notebooks
    /src       - modules
    /test      - unit tests
</pre>
