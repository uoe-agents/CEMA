#!/bin/bash

echo "Running experiments for scenarios and queries defined in 'final_queries.json'".

for sid in 1 2 3
do
  python scenarios/run.py $sid --query_path scenarios/queries/final_queries.json --save_causes
done

echo "Evaluating Scenario 1"
for i in 1 2 3
do
  python eval.py 1 $i
done

for sid in 2 3
do
  echo "Evaluation scenario $sid"
  for i in 1 2
  do
    python eval.py $sid $i
  done
done
