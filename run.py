import numpy as np 
from implementations import *
from helpers import *
from helper_functions import *
from optimization import *

# load the training dataset
print("loading the training dataset")
DATA_TRAIN_PATH = 'train.csv'
y_tr, tx_tr, ids_tr = load_csv_data(DATA_TRAIN_PATH)
print("training dataset loaded")

# Data cleaning 
print("preprocessing the data")
txOpti, yOpti, _ = dataClean(tx_tr, y_tr)
print("preprocessing done")

# Training with our best model: ridge regression 
ws = []
losses = []
txPoly = []

for i in range(4): 
    print(f"shape of txOpti[{i}] ", txOpti[i].shape)
    yOpti[i][np.where(yOpti == -1)] = 0
    initial_w = np.zeros(txOpti[i].shape[1])
    lambda_, degree = best_lambda_degree(yOpti[i], txOpti[i], 4, np.logspace(-6, 0, 30), np.arange(2,5), 1)
    t = build_poly(txOpti[i], degree)
    w, loss = ridge_regression(yOpti[i], t, lambda_)
    ws.append(w)
    losses.append(loss)
    txPoly.append(t)
    print("end of the ridge_regression with w=",w," and loss=", loss)
    
# Train accuracy
labels = []
accs = []
for i in range(4): 
    label = predict(ws[i], txPoly[i])
    yOpti[i][np.where(yOpti[i] == 0)] = -1
    acc = calculate_accuracy(yOpti[i], label)
    print("the accuracy on the train set is ", acc)
    accs.append(acc)
    labels.append(label)

accTot = (accs[0] + accs[1] + accs[2] + accs[3])/4
print("the total accuracy on the train set is ", accTot)

# Prediction 

# load the test data
print("loading the test dataset")
DATA_TEST_PATH = 'test.csv'
y_te, tx_te, ids_te = load_csv_data(DATA_TEST_PATH)
print("test dataset loaded")

# Data cleaning
txOpti_te, yOpti_te, idsOpti_te = dataClean(tx_te, y_te)

# Logistic prediction
label = np.zeros(len(ids_te))
for i in range(4):
    predicted = predict_logistic(ws[i], txOpti_te[i])
    
# Submission 
#OUTPUT_PATH = 'final-submission'
#create_csv_submission(idsOpti_te, predicted, OUTPUT_PATH)