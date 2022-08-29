import json
import pdb
import os
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

#%matplotlib inline


#load data line by line
data = []
for line in open(os.getcwd() + '/Data/transactions.txt', 'r'):
    data.append(json.loads(line))

#printing the size of data
print("number of records : ", len(data))
print("number of fields in each record : ", len(data[0]))
print("fields : ", data[0].keys())

#Converting to pandas dataframe
data_df = pd.DataFrame(data)

#checking for null values
print(data_df.isnull().sum())

#Describing data
print(data_df.describe())

#iterate over df coloumns
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

#pdb.set_trace()

data_df.hist(column='transactionAmount', bins=30)
plt.show()

#pdb.set_trace()
