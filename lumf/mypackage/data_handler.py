import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def __fillna_and_scale_numberic_features(df: pd.DataFrame):
    features = df.dtypes[df.dtypes != 'object'].index
    for ft_name in features:
        has_null = df[ft_name].isnull().sum() > 0
        if has_null:
            ft_mean = df[ft_name].mean()
            df[ft_name].fillna(ft_mean, inplace=True)
    df[features] = StandardScaler().fit_transform(df[features])
    return df


def __fillna_and_dummy_category_feature(df: pd.DataFrame):
    features = df.dtypes[df.dtypes == 'object'].index
    for ft_name in features:
        has_null = df[ft_name].isnull().sum() > 0
        if has_null:
            # 众数
            ft_mode = df[ft_name].mode()[0]
            df[ft_name].fillna(ft_mode, inplace=True)
    df = pd.get_dummies(df, columns=features)
    return df


def get_train_X(df: pd.DataFrame):
    df = __fillna_and_scale_numberic_features(df)
    df = __fillna_and_dummy_category_feature(df)
    return df


def get_test_X(df: pd.DataFrame):
    df = __fillna_and_scale_numberic_features(df)
    df = __fillna_and_dummy_category_feature(df)
    return df

def match_columns_for_text_X(test_X: pd.DataFrame, train_X : pd.DataFrame):
    miss_columns = set(train_X.columns) - set(test_X.columns)
    # add missing dummy columns
    for col in miss_columns:
        test_X[col] = 0
    adu_columns = set(train_X.columns) - set(train_X.columns)
    # delete adundant columns:
    test_X.drop(list(adu_columns), axis=1, inplace=True)
    return test_X
