## ML project 1

### Team

This repository contains the code for the project1 of the Machine Learning Course. The team is composed of

   - Ajkuna Seipi (ajkuna.seipi@epfl.ch)
   - Hongyi Shi (hongyi.shi@epfl.ch)
   - Louis Perrier (louis.perrier@epfl.ch)

# Project structure
## Datas : 
To run our code, download the data from [AIcrowd | EPFL Machine Learning Higgs | Challenges](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs)  and put it at the root of our project. Otherwise, the user can simply change the data paths in `run.py` file to their own paths. 
## Files : 
`optimization.py` : process data for model by spliting the classes into four groups, replace the -999 values by the NaN value, remove the outliers, and standardize our dataset. 
`helpers.py`: the provided helper functions to load the datas and create a submission file.
`helper_functions.py` : the functions we needed to implement the methods in `implementations.py`. We also added two functions that predict the label given the weights and the dataset.
`implementations.py`: the implementation of our 6 methods to train the dataset. 
`optimization.py` : process data for model by spliting the classes into four groups, replace the -999 values by the NaN value, remove the outliers, and standardize our dataset. 
`report.pdf`: a 2-pages report of the complete solution, which describes the whole procedure of our findings.
`run.py`: the results using our best model. 
`finalsubmission.csv`: the predictions for the test data with our best model.

## Running the code : 
You need to move to the root folder and execture ` python run.py`. 