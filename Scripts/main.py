import json
import pdb
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import load_data, preprocess_data, preprocess_data_for_fraud_model, process_data_for_fraud_model
from data_wrangling import reverse_transactions, duplicate_transactions

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

import argparse

def main():

    #arguments
    parser = argparse.ArgumentParser(description='Options for running some steps')
    parser.add_argument('--reverse_transactions', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--duplicate_transactions', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--train', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    #load data, saved in a list
    data_path = '/Data/transactions.txt'
    data = load_data(data_path)

    #preprocess data
    data_preprocessed_df = preprocess_data(data)

    #plot transaction amount values
    data_preprocessed_df.hist(column='transactionAmount', bins=100)
    plt.show()

    #identify reversed transactions
    if args.reverse_transactions:
        total_reverse_transactions, total_dollar_amount_in_reversed_transactions, reverse_transactions_details_df = reverse_transactions(data_preprocessed_df)
        print(" Total number of reverse transcations: {}  and total amount of reverse transactions: {} $".format(total_reverse_transactions, total_dollar_amount_in_reversed_transactions))

    #identify duplicate transactions
    if args.duplicate_transactions:
        total_duplicate_transactions, total_dollar_amount_in_duplicate_transactions = duplicate_transactions(data_preprocessed_df)
        print(" Total number of duplicate transcations: {}  and total amount of duplicate transactions: {} $".format(total_duplicate_transactions, total_dollar_amount_in_duplicate_transactions))

    #Data processing for fraud detection model
    data_preprocessed_df = preprocess_data_for_fraud_model(data_preprocessed_df)

    #Fraud detection
    x_train, x_test, y_train, y_test = process_data_for_fraud_model(data_preprocessed_df)

    if args.train:
        params = {'penalty': ['l2', 'none'],
            'C': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 200],
            'fit_intercept': [True, False],
            'n_jobs': [-1]
        }

        grid = GridSearchCV(estimator=LogisticRegression(max_iter=2000), param_grid=params, cv=5, verbose=3)
        grid.fit(x_train, y_train)

        log_regresstion_model = grid.best_estimator_

        with open('model/logistic_regresstion_model', 'wb') as file:
            pickle.dump(log_regresstion_model, file)
    else:
        with open('model/logistic_regresstion_model', 'rb') as file:
            log_regresstion_model = pickle.load(file)


    y_prediction = log_regresstion_model.predict(x_test)
    y_prediction_prob = log_regresstion_model.predict_proba(x_test)[:,1]
    y_true = y_test.tolist()

    confusion_matrix_val = confusion_matrix(y_true, y_prediction)
    accuracy_score_val = accuracy_score(y_true, y_prediction)

    print("True Positive: ", confusion_matrix_val[1,1])
    print("True Negative: ", confusion_matrix_val[0,0])
    print("False Negative: ", confusion_matrix_val[1,0])
    print("False Positive: ", confusion_matrix_val[0,1])

    tp = confusion_matrix_val[1,1]
    tn = confusion_matrix_val[0,0]
    fn = confusion_matrix_val[1,0]
    fp = confusion_matrix_val[0,1]

    precision_val = tp/(tp+fp)
    recall_val = tp/(tp+fn)

    print(" Precision :", precision_val)
    print(" Recall :", recall_val)
    print(" Accuracy :", accuracy_score_val)

    pdb.set_trace()

if __name__ == '__main__':
    main()
