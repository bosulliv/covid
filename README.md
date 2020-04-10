# Covid
## Overview
Explore the covid data. Fit simple models. Update as daily figures arrive, and reflect on the changes necessary to improve the model fit.

## Country Models
Individual countries are modeled and ploted here:

* [Germany](notebooks/Germany.ipynb)
* [Italy](notebooks/Italy.ipynb)
* [Spain](notebooks/Spain.ipynb)
* [UK](notebooks/uk.ipynb)
* [US](notebooks/US.ipynb)


[This notebook](notebooks/covid.ipynb) is the original notebook, it's a bit messy, but has more countries at present. It is soon to be depreciated as the code has been refactored and moved into a module.


## Project Structure
<pre>
    /data      - virus time series, country metadata, and model paramater values
    /notebooks - notebooks
    /src       - modules
    /test      - unit tests
</pre>

## Download and Run
Clone the code:
<pre>
$ git clone git@github.com:bosulliv/covid.git
$ cd covid
</pre>

I recommend conda to manage the libraries and create an environment for this repo.  [Install miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html) if you don't have conda yet. It's a minimal install, and this repo uses a small number of libraries.

<pre>
(base) $ conda create --name covid --file spec-file.txt
(base) $ conda activate covid
(covid) $ 
</pre>

Then run jupyter to explore the notebooks and raw data files:
<pre>
(covid) $ jupyter notebook
</pre>