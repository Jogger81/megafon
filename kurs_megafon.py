import pandas as pd
import dask.dataframe as dd
import numpy as np
import pickle

import sys
import argparse

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# параметры командной строки:
# -m    имя файла сохраненной модели (по умолчанию model.sav)
# -t    имя файла для классификации (по умолчанию data_test.sav)
# -f    имя файла features (по умолчанию features.sav)
# -dest имя файла для сохранения результатоа (по умолчанию answers_test.sav)
# -d    рабочая директория (по умолчанию текущая)
#


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', dest="model", default='model.sav')
    parser.add_argument('-t', '--test', dest="data_test", default='data_test.csv')
    parser.add_argument('-f', '--features', dest="data_features", default='features.csv')
    parser.add_argument('-dest', '--destination', dest="destination", default='answers_test.csv')
    parser.add_argument('-d', '--directory', dest="filepath", default='./')
    return parser

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("DataFrame не содердит следующие колонки: %s" % cols_error)

if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    print('-'*5+'Полученные параметры'+'-'*5)
    print('model = '+namespace.model)
    print('test file = '+namespace.data_test)
    print('features file = '+namespace.data_features)
    print('file path = '+namespace.filepath)
    print('destination = '+namespace.destination)
    print('-' * 30)

    print('Обработка Data_test 1/7')
    data_test = dd.read_csv(namespace.filepath+namespace.data_test).drop(columns=['Unnamed: 0'])
    data_test = data_test.sort_values(['id']).reset_index().drop(columns=['index'])

    print('Обработка features 2/7')
    df = dd.read_csv(namespace.filepath+namespace.data_features, delimiter="\t").drop(columns=['Unnamed: 0'])
    df = df.sort_values(['id']).reset_index().drop(columns=['index'])

    print('Data_test join with features 3/7')
    X_test = data_test.join(df, on="id", rsuffix='_other').compute()

    X_nunique = X_test.apply(lambda x: x.nunique(dropna=False))
    f_all = set(X_nunique.index.tolist())

    if isinstance(X_test, pd.DataFrame):
        print('Загружаем модель 4/7')
        loaded_model = pickle.load(open(namespace.filepath + namespace.model, 'rb'))

        print('Получаем предсказание 5/7')
        model_pred = loaded_model.predict_proba(X_test)

        print('Подготавливаем данные 6/7')
        X_test = X_test[['buy_time', 'id', 'vas_id']]
        X_test['target'] = model_pred[:,1]

        print('Сохраняем в файл 7/7')
        X_test.to_csv(namespace.filepath + namespace.destination)

        print('ГОТОВО!')
    else:
        print('ОШИБКА ДАТАСЕТА')



