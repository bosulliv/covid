#!/bin/bash
# TODO: This needs to work in all cron's - not just mine
. covid_env/bin/activate

compfile=./notebooks/Comparison.ipynb
skewfilw=./notebooks/Skew\ Function.ipynb
t_o=300

for file in $(ls ./notebooks/*.ipynb)
do
  if [[ $file != $compfile && $file != $skewfile ]] 
  then
    echo $file
    ~/anaconda3/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=$t_o --inplace $file
  fi
done

# Do the comparison after all files
~/anaconda3/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=$t_o --inplace $compfile

git add .
git commit -m 'cron update'
git push

deactivate