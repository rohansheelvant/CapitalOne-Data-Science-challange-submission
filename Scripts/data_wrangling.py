import json
import pdb
import os
import pandas as pd

def reverse_transactions(data_preprocessed_df):
    # Function to calculate total number of reverse transactions and total amount in reverse transactions
    total_reverse_transactions = 0
    total_dollar_amount_in_reversed_transactions = 0
    reverse_transactions_details = []
    for row_index, row_value in data_preprocessed_df.T.iteritems():
        print(row_index)
        if row_value['transactionType'] == 'REVERSAL':
            account_number = row_value['accountNumber']
            transaction_amount = row_value['transactionAmount']
            transaction_matches = data_preprocessed_df.query(' accountNumber=="{}" & transactionAmount=={} '.format(account_number, transaction_amount))
            for transaction_index, transaction_value in transaction_matches.T.iteritems():
                if( transaction_value['transactionDateTime'] < row_value['transactionDateTime'] ):
                    reverse_transactions_details.append(transaction_value)
                    reverse_transactions_details.append(row_value)
                    total_reverse_transactions = total_reverse_transactions + 1
                    total_dollar_amount_in_reversed_transactions = total_dollar_amount_in_reversed_transactions + transaction_amount


    return total_reverse_transactions, total_dollar_amount_in_reversed_transactions, pd.DataFrame(reverse_transactions_details)


def duplicate_transactions(data_preprocessed_df):
    # Function to calculate total number of duplicate transactions and total amount in duplicate transactions
    total_duplicate_transactions = 0
    total_dollar_amount_in_duplicate_transactions = 0
    for row_index, row_value in data_preprocessed_df.T.iteritems():
        print(row_index)
        if row_value['transactionType'] == 'PURCHASE':
            account_number = row_value['accountNumber']
            transaction_amount = row_value['transactionAmount']
            transaction_matches = data_preprocessed_df.query(' accountNumber=="{}" & transactionAmount=={} '.format(account_number, transaction_amount))
            for transaction_index, transaction_value in transaction_matches.T.iteritems():
                if( row_value['transactionDateTime'] - transaction_value['transactionDateTime'] < pd.Timedelta(120, unit='s') and row_value['transactionDateTime'] - transaction_value['transactionDateTime'] > pd.Timedelta(0, unit='s')  ):
                    total_duplicate_transactions = total_duplicate_transactions + 1
                    total_dollar_amount_in_duplicate_transactions = total_dollar_amount_in_duplicate_transactions + transaction_amount

    return total_duplicate_transactions, total_dollar_amount_in_duplicate_transactions
