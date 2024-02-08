import pandas as pd
#from google.cloud import bigquery
import numpy as np
import matplotlib as plt
from pathlib import Path

#pip install category_encoders
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from category_encoders import TargetEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

def clean_data(df_dvf: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - removing irrelevant columns
    - choose relevant values inside columns
    - assigning correct dtypes to each column
    - remove NaN and dupliactes
    returns a dataframe with cleaned data
    """

    #removing columns --> TO DO: should this be in params?
    keep_col =['date_mutation',
            'nature_mutation',
            'valeur_fonciere',
            'code_postal',
            'code_commune',
            'code_departement',
            'nombre_lots',
            'type_local',
            'surface_reelle_bati',
            'nombre_pieces_principales',
            'longitude',
            'latitude']
    df_dvf = df_dvf[keep_col]

    # TO DO : keep surface_terrain -> NaN => 0 (gives an indication if there is a garden or not)

    #translate the columns
    df_dvf.columns = ['date', 'built', 'price', 'postal_code',
                'city', 'region', 'number_of_units', 'property_type',
                'living_area', 'number_of_rooms',
                'longitude', 'latitude']

    #filter values inside columns:
        # 1.Keeping only regular sales,
        # 2.Filter for transactions for 1 unit,
        # 3.Only consider property types for living.
    df_useful = df_dvf[((df_dvf['built'] == "Vente") | (df_dvf['built'] == "Vente en l'état futur d'achèvement")) &
                    ((df_dvf['number_of_units'] == 1) | (df_dvf['number_of_units'] == '1')) &
                    ((df_dvf['property_type'] == 'Appartement') | (df_dvf['property_type'] == 'Maison'))]

    #translate values
    trans_dict_built = {'Vente' : 'built',
                        "Vente en l'état futur d'achèvement" : 'off-plan'}
    trans_dict_type = {'Appartement' : 'appartment',
                    'Maison' : 'house'}
    df_useful= df_useful.replace({'built' : trans_dict_built,
                                'property_type' : trans_dict_type})

    #dropping the column with number of units (only 1s)
    df_useful = df_useful.drop(columns='number_of_units')


    #checking for NaN and duplicate values
    df_useful = df_useful.dropna()
    df_useful = df_useful.drop_duplicates()

    #changing evething to the right type --> TO DO:should this go to params?
    col_float = ['price', 'longitude', 'latitude', 'living_area', 'number_of_rooms']
    col_string = ['built','city', 'region','property_type']
    col_date = ['date']
    col_int = ['postal_code']
    #formating data types
    df_useful[col_float] = df_useful[col_float].apply(lambda x: pd.to_numeric(x, errors='coerce').astype('float64'))
    df_useful[col_date]= df_useful[col_date].apply(lambda x: pd.to_datetime(x, errors='coerce'))
    df_useful[col_int]= df_useful[col_int].apply(lambda x: pd.to_numeric(x, errors='coerce').astype('int64'))

    return df_useful

# def filter_data(max_price, max_area):


#get data
def preprocess_data(df_clean : pd.DataFrame, robust = True) -> pd.DataFrame:
    """ The preprocess_data function is splitting the clean data into
    X_train_preproc, X_test_preproc, y_train and y_test
    """
    X= df_clean.drop(columns=['price'])
    y= df_clean['price']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3)

    # Preprocessing numerical data
    numeric_transformer = Pipeline([
        ('minmax', (RobustScaler() if robust == True else MinMaxScaler()))
        ])

    # Preprocessing categorical data
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(sparse_output=False, dtype=int))
        ])

    # Preprocesing target encoder
    targetencoder_transformer = Pipeline([
        ('target_enc', TargetEncoder())
        ])

    # Parallelize "num_transformer" and "cat_transfomer"
    preprocessor = ColumnTransformer([
            ('num', numeric_transformer, ['living_area', 'number_of_rooms', 'longitude', 'latitude']),
            ('cat', categorical_transformer, ['property_type']),
            ('tar', targetencoder_transformer, ['city', 'postal_code'])
            ])

    # preprocessing pipeline
    preprocessing_pipeline = Pipeline([('preprocessor', preprocessor)])

    # Apply  pipeline to  dataset AND keep column names
    X_train_preproc_ = preprocessing_pipeline.fit_transform(X_train, y_train)
    X_train_preproc = pd.DataFrame(X_train_preproc_, columns = preprocessing_pipeline.get_feature_names_out(X_train.columns))
    X_test_preproc_ = preprocessing_pipeline.transform(X_test)
    X_test_preproc = pd.DataFrame(X_test_preproc_, columns = preprocessing_pipeline.get_feature_names_out(X_test.columns))

    y_train = pd.DataFrame(y_train, columns = ['price'])
    y_test = pd.DataFrame(y_test, columns = ['price'])

    X_all = pd.concat([X_train_preproc, X_test_preproc], axis=0, ignore_index=True)
    y_all = pd.concat([y_train, y_test], axis=0, ignore_index=True)

    col = list(X_all.columns) + ['price']
    df_full = pd.concat([X_all, y_all], axis = 1, names = col)

    return X_train_preproc, X_test_preproc, y_train, y_test, X_all, y_all, df_full
