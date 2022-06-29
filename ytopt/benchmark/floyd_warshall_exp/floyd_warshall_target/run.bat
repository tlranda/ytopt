python -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=200 --learner RF  
cp results.csv medium-rf.csv
python -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=200 --learner GP  
cp results.csv medium-gp.csv
python -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=200 --learner GBRT  
cp results.csv medium-gbrt.csv
python -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=200 --learner ET  
cp results.csv medium-et.csv
python findMin.py

