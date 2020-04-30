#!/bin/bash
compfile=./notebooks/Comparison.ipynb
skewfile=./notebooks/SkewFunction.ipynb
t_o=360

for file in $(ls ./notebooks/*.ipynb)
do
  if [[ $file != $compfile && $file != $skewfile ]] 
  then
    echo $file
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=$t_o --inplace $file
  fi
done

# Do the comparison after all files
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=$t_o --inplace $compfile

deactivate
