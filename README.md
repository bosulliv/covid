[![openssf scorecards](https://api.securityscorecards.dev/projects/github.com/coreinfrastructure/best-practices-badge/badge)](https://api.securityscorecards.dev/projects/github.com/coreinfrastructure/best-practices-badge)
[![OpenSSF Scorecard](htt‌ps://api.securityscorecards.dev/projects/github.com/bosulliv/covid/badge)](htt‌ps://securityscorecards.dev/viewer/?uri=github.com/bosulliv/covid)
[![Scorecard supply-chain security](https://github.com/bosulliv/covid/actions/workflows/scorecard.yml/badge.svg)](https://github.com/bosulliv/covid/actions/workflows/scorecard.yml)

# Covid
## Overview
Explore the covid data. Fit simple models. Update as daily figures arrive, and reflect on the changes necessary to improve the model fit.

## Country Models
My home country, UK, is explored in the most detail. Including day of week patterns (mid-week = low, weekend = high):
* [UK](notebooks/uk.ipynb)

Other countries are modeled and ploted here:
* [Australia](notebooks/Australia.ipynb)
* [Brazil](notebooks/Brazil.ipynb)
* [France](notebooks/France.ipynb)
* [Germany](notebooks/Germany.ipynb)
* [India](notebooks/India.ipynb)
* [Italy](notebooks/Italy.ipynb)
* [Netherlands](notebooks/Netherlands.ipynb)
* [Romania](notebooks/Romania.ipynb)
* [Russia](notebooks/Russia.ipynb)
* [Singapore](notebooks/Singapore.ipynb)
* [Spain](notebooks/Spain.ipynb)
* [Sweden](notebooks/Sweden.ipynb)
* [US](notebooks/US.ipynb)

These countries are then compared here:
* [Comparison](notebooks/Comparison.ipynb)

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

You can update all the notebooks at once with the repo shell script. It will take about 10 minutes:
<pre>
(covid) $ chmod a+x src/run_all.sh
(covid) $ ./src/run_all.sh
</pre>

If you want to add this to cron, the provided venv environment works much better with cron. Grafting conda into a shell started by cron is not elegant of robust. Here is my crontab, on OSX:

<pre>
$ crontab -l
SHELL=/bin/bash
1 7 * * * cd /Users/Brian/Documents/Code/python/covid ; ./src/run_all_cron.sh
1 20 * * * cd /Users/Brian/Documents/Code/python/covid ; ./src/run_all_cron.sh
</pre>

## The Maths
This is curve fitting, rather than virus transmission simulation. Two curves have been used, Sigmoid and Gamma. I started with sigmoid because it is tuned with a single parameter, which makes it simple. This was helpful in early March, because it was too early to know what a 'typical' curve might look like - and therefore it is easy to overfit with more complicated models when you are training it with a fraction of the expected data points.

Once countries pass their peak, it is clear the curves are steep on the climb and much less steep on the descent. This is called skew. The sigmoid is a symetrical version of the cumulative Gamma function. But if we use a pure Gamma distribution function, we can also tune the skew. Now we have a better idea of what the 'typical' can look like, we can use a function with more degrees of freedom.

This function fits much better, and is the now the default curve to fit.
