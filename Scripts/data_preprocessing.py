import json
import pdb
import os
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler


def load_data(path):
    #loads data from the path and saves it as a list
    data = []
    for line in open(os.getcwd() + path, 'r'):
        data.append(json.loads(line))

    return data

def preprocess_data(data):
    # convert to pandas dataframe
    data_df = pd.DataFrame(data)
    data_df.replace('', np.nan, inplace=True)
    print("Number of nan values : ")
    print(data_df.isnull().sum())

    #printing the size of data
    print("number of records : ", len(data))
    print("number of fields in each record : ", len(data[0]))
    print("Name of fields : ", data[0].keys())


    #iterate over df coloumns and print stats
    for key, value in data_df.iteritems():
        print(" Null values in "+ key + "  :  ", value.isnull().sum())
        try:
            print(" Max value in " + key + "  :  ", value.max())
            print(" Min value in " + key + " : ", value.min())
        except:
            print(" Max & Min values cant be compututed for this field")
        print(" Number of unique values : ", value.nunique())
        print()


    #Dropping dataa will all null values
    data_df.drop(['merchantCity', 'merchantState', 'merchantZip', 'posOnPremises', 'recurringAuthInd'], axis=1, inplace=True)
    data_df.isnull().sum()
    data_df = data_df.reset_index(drop=True)

    #converting data time column in DateTime data type
    data_df['transactionDateTime'] = pd.to_datetime(data_df['transactionDateTime'])

    return data_df


def preprocess_data_for_fraud_model(data_preprocessed_df):
    data_preprocessed_df['matchingCVV'] = data_preprocessed_df['cardCVV'] == data_preprocessed_df['enteredCVV']

    for col in ['cardPresent', 'matchingCVV', 'expirationDateKeyInMatch', 'isFraud']:
        data_preprocessed_df[col] = data_preprocessed_df[col].replace({False: 0, True: 1})

    data_preprocessed_df.drop(['echoBuffer', 'cardLast4Digits',
         'merchantName', 'accountOpenDate',
         'transactionDateTime', 'currentExpDate',
         'customerId','dateOfLastAddressChange',
         'accountNumber','enteredCVV','cardCVV'], inplace=True, axis=1)
    return data_preprocessed_df


def process_data_for_fraud_model(data_preprocessed_df):
    random_sampler = RandomUnderSampler()
    y = data_preprocessed_df['isFraud']
    data_preprocessed_df.drop('isFraud', inplace=True, axis=1)
    X, Y = random_sampler.fit_resample(data_preprocessed_df, y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y)

    pipeline = ColumnTransformer([
    ('cat_pipe', Pipeline([
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot', OneHotEncoder(handle_unknown='ignore'))]),
     ['merchantCountryCode','merchantCategoryCode','posConditionCode','posEntryMode','transactionType','acqCountry']),
    ], remainder='passthrough')

    x_train = pipeline.fit_transform(x_train)
    x_test = pipeline.transform(x_test)

    return x_train, x_test, y_train, y_test
