import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from IPython.display import display, HTML


class DataPrinter:

    #参考https://zhuanlan.zhihu.com/p/66050679

    #计算缺失值比例
    @staticmethod
    def print_types_and_nullnum(df : pd.DataFrame):
        print('sample num:', df.shape[0])
        features_category = df.dtypes[df.dtypes == 'object'].index
        features_numberic = df.dtypes[df.dtypes != 'object'].index
        df_null_info = pd.DataFrame([df.dtypes, df.isnull().sum()], index=['type', 'na_sum'])
    
        display(df_null_info[features_category])
        display(df_null_info[features_numberic])

    @staticmethod
    def print_feature_info(df : pd.DataFrame, ft_name : str):
        print('\n-------------------------------------')
        print('feature name:', ft_name)
        print('feature_type:', df.dtypes[ft_name])
        #众数
        ft_mode = df[ft_name].mode()[0]
        print('feature_mode:', ft_mode)
        
        sample_count = df[ft_name].shape[0]
        na_count = df[ft_name].isnull().sum()
        print("na_count={}, na_count_percent={}".format(na_count, na_count / sample_count))
        
        if df.dtypes[ft_name] == 'object':
            print("value_counts----------start")
            print(df[ft_name].value_counts())
            print("value_counts----------end")

    @staticmethod        
    def __fillna_feature(df: pd.DataFrame, ft_name:str):
        has_null = df[ft_name].isnull().sum() > 0
        if df.dtypes[ft_name] == 'object':
            if has_null:
                ft_mode = df[ft_name].mode()[0]
                df[ft_name].fillna(ft_mode, inplace = True)
        else:
            if has_null:
                ft_mean = df[ft_name].mean()
                df[ft_name].fillna(ft_mean, inplace = True)

    @staticmethod            
    def __label_encode_feature(df: pd.DataFrame, ft_name: str):
        if df.dtypes[ft_name] == 'object':
            df[ft_name] = LabelEncoder().fit_transform(df[ft_name])
        return df
                    

    @staticmethod
    def __scale_numberic_feature(df : pd.DataFrame):
        features_numberic = df.dtypes[df.dtypes != 'object'].index
        df[features_numberic] = StandardScaler().fit_transform(df[features_numberic])
        
    @staticmethod
    def __get_hightly_corr(df, y:str, corr_threshold):
        corr_matrix = df.corr()[y]
        features = df.columns[corr_matrix.abs() > corr_threshold]
        return features
    
    @staticmethod
    def print_heatmap(df: pd.DataFrame):
        corr_matrix = df.corr()
        plt.figure()
        sns.heatmap(corr_matrix)

    @staticmethod
    def print_highly_related_features(df: pd.DataFrame, y:str, corr_threshold:float):
        for cl in df.columns:
            DataPrinter.__fillna_feature(df, cl)
                
        for cl in df.columns:
            DataPrinter.__label_encode_feature(df, cl)
        # DataPrinter.print_types_and_nullnum(df)
        highly_features = DataPrinter.__get_hightly_corr(df, y, corr_threshold)
        print("\nhighly corr features---------corr_threshold={}".format(corr_threshold))
        print(highly_features)


