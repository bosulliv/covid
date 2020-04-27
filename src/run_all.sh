#!/bin/bash
compfile=./notebooks/Comparison.ipynb
skewfilw=./notebooks/Skew\ Function.ipynb
to=180

for file in $(ls ./notebooks/*.ipynb)
do
  if [[ $file != $compfile && $file != $skewfile ]] 
  then
    echo $file
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=$to --inplace $file
  fi
done

# Do the comparison after all files
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=$to --inplace $compfile

deactivate