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
print("training the model")
degree = 4
losses = []
ws = []
txPoly = [] 

for i in range(4): 
    yOpti[i][np.where(yOpti[i] == -1)] = 0

    t = build_poly(txOpti[i], degree)
    w, loss = ridge_regression(yOpti[i], t, 0.000001)
    losses.append(loss) 
    ws.append(w) 
    txPoly.append(t)
    print(f'end of the ridge_regression with loss={loss}')
    
    #accuracy
    predicted = predict(w, t)
    yOpti[i][np.where(yOpti[i] == 0)] = -1
    acc = calculate_accuracy(yOpti[i], predicted)
    print(f'{acc:.15f}')
    
# Train accuracy 
print("accuracy of our model on the test set")
y_pred = np.zeros(len(ids_tr))
degree = 4

for i in range(4):
    predicted = predict(ws[i], txPoly[i])
    y_pred[idsOpti[i]] = predicted

y_pred[np.where(y_pred == 0)] = -1

acc = calculate_accuracy(y_tr, y_pred)
print(f'{acc:.15f}')

# Prediction 

# load the test data
print("loading the test dataset")
DATA_TEST_PATH = 'test.csv'
y_te, tx_te, ids_te = load_csv_data(DATA_TEST_PATH)
print("test dataset loaded")

# Data cleaning of the test set
print("preprocessing the test set")
txOpti_te, yOpti_te, idsOpti_te = dataClean(tx_te, y_te)

# Label prediction
y_pred_te = np.zeros(len(ids_te))
degree = 4

for i in range(4):
    t = build_poly(txOpti_te[i], degree)
    predicted = predict(ws[i], t)
    y_pred_te[idsOpti_te[i]] = predicted
    
# Submission 
OUTPUT_PATH = 'finalsubmission'
create_csv_submission(idsOpti_te, predicted, OUTPUT_PATH)