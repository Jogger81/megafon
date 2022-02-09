import pandas as pd
import dask.dataframe as dd
import numpy as np

import sys
import argparse


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', dest="model", required=True)
    parser.add_argument('-t', '--test', dest="data_test", required=True)
    parser.add_argument('-d', '--directory', dest="filepath", default='./')
    return parser


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    print(namespace)

    for name in namespace.name.split():
        print("Привет, {}!".format(name))




    data_test = dd.read_csv('data_train.csv')
    data_train = data_train.set_index('Unnamed: 0')
    data_train.to_hdf('file.hdf5', key='table')
    data_train = data_train.sort_values('id')
    print(data_train.shape)
    print(data_train.head())

    df = dd.read_csv('features.csv', delimiter="\t")
    df = df.set_index('Unnamed: 0')
    df = df.sort_values('id')
    df.to_hdf('file.hdf5', key='table')
    print(df.head())

    data = dd.multi.merge_asof(data_train, df, on="buy_time", by="id")
    data.to_csv('data_with_features.csv')

    print(data.head(3))