# Covid
## Overview
Explore the covid data. Fit simple models. Update as daily figures arrive, and reflect on the changes necessary to improve the model fit.

## Country Models
Individual countries are modeled and ploted here:
* [Germany](notebooks/Germany.ipynb)
* [Italy](notebooks/Italy.ipynb)
* [Spain](notebooks/Spain.ipynb)
* [Sweden](notebooks/Spain.ipynb)
* [UK](notebooks/uk.ipynb)
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

## The Maths
This is curve fitting, rather than virus transmission simulation. Two curves have been used, sigmoid and gamma. I started with sigmoid because it is very simple and just has a single parameter. This was helpful in early March, because it was too early to know what a 'typical' curve might look like.

However, some Countries are now passed their daily peak and it's clear the curve is skewed. A gamma function has two paramaters, one of which can tune the skew. Now we have a better idea of what 'typical' can look like, we can use a function with more degrees of freedom.

This function fits much better, and is the now the default curve to fit.