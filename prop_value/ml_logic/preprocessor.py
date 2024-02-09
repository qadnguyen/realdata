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
import pickle

import warnings
warnings.filterwarnings('ignore')

def clean_data(df_dvf: pd.DataFrame, percentile = 0.95) -> pd.DataFrame:
    """
    Clean raw data by
    - removing irrelevant columns
    - choose relevant values inside columns
    - assigning correct dtypes to each column
    - remove NaN and dupliactes
    returns a dataframe with cleaned data
    """

    #removing columns --> TO DO: should this be in params?
    keep_col =['id_mutation',
                'date_mutation',
                'nature_mutation',
                'valeur_fonciere',
                'code_postal',
                'code_commune',
                'code_departement',
                'type_local',
                'surface_reelle_bati',
                'nombre_pieces_principales',
                'surface_terrain',
                'longitude',
                'latitude']
    df_dvf = df_dvf[keep_col]

    # TO DO : keep surface_terrain -> NaN => 0 (gives an indication if there is a garden or not)

    #translate the columns
    df_dvf.columns = ['id', 'date', 'built', 'price', 'postal_code',
                    'city', 'region', 'property_type',
                    'living_area', 'number_of_rooms', 'land_area',
                    'longitude', 'latitude']

    #filter values inside columns:
        # 1.Keeping only regular sales,
        # 2.Filter for transactions for 1 unit,
        # 3.Only consider property types for living.
    df_useful = df_dvf[((df_dvf['built'] == "Vente") | (df_dvf['built'] == "Vente en l'état futur d'achèvement")) &
                        ((df_dvf['property_type'] == 'Appartement') | (df_dvf['property_type'] == 'Maison')  | (df_dvf['property_type'] == 'Maison')| (df_dvf['property_type'] == 'Dépendance'))]

    #translate values
    trans_dict_built = {'Vente' : 'built',
                        "Vente en l'état futur d'achèvement" : 'off-plan'}
    trans_dict_type = {'Appartement' : 'appartment',
                    'Maison' : 'house',
                    'Dépendance' : 'dep'}
    df_useful= df_useful.replace({'built' : trans_dict_built,
                             'property_type' : trans_dict_type})

    #add new columns to eventually filter out sales with different units
    ## 0. Creating the masks
    mask_house = df_useful.property_type == 'house'
    mask_app = df_useful.property_type == 'appartment'
    mask_dep = df_useful.property_type == 'dep'
    ## 1. find the number of houses per id
    num_of_houses_df = df_useful[['id','property_type']][mask_house].groupby(by = 'id', as_index = False).count()
    num_of_houses_df.columns = ['id', 'nb_of_house']
    ## 2. find the number of appartments per id
    num_of_app_df = df_useful[['id','property_type']][mask_app].groupby(by = 'id', as_index = False).count()
    num_of_app_df.columns = ['id', 'nb_of_app']

    ## 3. find the number of dep per id
    num_of_dep_df = df_useful[['id','property_type']][mask_dep].groupby(by = 'id', as_index = False).count()
    num_of_dep_df.columns = ['id', 'nb_of_dep']

    ## 4. Merging 3 columns with df_useful
    df_feat = df_useful.merge(num_of_houses_df, on = 'id', how = 'outer')
    df_feat = df_feat.merge(num_of_app_df, on = 'id', how = 'outer')
    df_feat = df_feat.merge(num_of_dep_df, on = 'id', how = 'outer')

    # fill out the blanks for the 3 new columns + land_area
    df_feat.fillna({'nb_of_house' : 0, 'nb_of_app' : 0, 'nb_of_dep' : 0, 'land_area' : 0}, inplace=True)

    # remove lines with multiple houses or appartments in same sale_id
    df_feat['nb_of_units'] = df_feat.nb_of_app+df_feat.nb_of_house
    mask_units = (df_feat['nb_of_units'] == 1)
    df_filtered = df_feat[mask_units]

    #filter out the dependencies
    filter_type = (df_filtered['property_type'] != 'dep')
    df_filtered = df_filtered[filter_type]

    # drop the useless columns
    df_filtered_drop = df_filtered.drop(columns = ['nb_of_units', 'nb_of_app', 'nb_of_house'])

    # drop na + duplicates (shouldn't be duplicates anymore at this point)
    df_final = df_filtered_drop.drop(columns = ['land_area', 'id'])
    df_clean = df_final.dropna()

    # Create price per m2
    df_clean[['price', 'living_area']] = df_clean[['price', 'living_area']].apply(lambda x: pd.to_numeric(x, errors='coerce').astype('float64'))

    df_clean['price_per_m2'] = df_clean['price'] / df_clean['living_area']

    #changing evething to the right type
    col_float = ['price', 'longitude', 'latitude', 'living_area', 'price_per_m2' ]
    col_string = ['built','city', 'region','property_type']
    col_date = ['date']
    col_int = ['postal_code', 'nb_of_dep', 'number_of_rooms']
    #formating data types
    df_clean[col_float] = df_clean[col_float].apply(lambda x: pd.to_numeric(x, errors='coerce').astype('float64'))
    df_clean[col_date]= df_clean[col_date].apply(lambda x: pd.to_datetime(x, errors='coerce'))
    df_clean[col_int]= df_clean[col_int].apply(lambda x: pd.to_numeric(x, errors='coerce').astype('int64'))

    ##### FILTER OUT OUTLIERS WITH PRICE PER M2 ABOVE P95 (or other if mentioned)
    # finding the percentile 95 for each postcode
    df_quantile = df_clean[['postal_code', 'price_per_m2']].groupby(by = df_clean.postal_code, as_index=False).quantile(percentile)
    df_quantile.columns = ['postal_code', 'p95']

    # merge the df
    df_clean_merged = df_quantile.merge(right = df_clean, on = 'postal_code')

    # drop the outliers
    mask_outliers = (df_clean_merged.price_per_m2 < df_clean_merged.p95)
    df_without_outliers = df_clean_merged[mask_outliers]

    # drop useless columns
    df_without_outliers = df_without_outliers.drop(columns=['p95', 'price_per_m2'])

    #change strings in city to integers for a common city code
    df_without_outliers['city'] = pd.to_numeric(df_without_outliers['city'], errors='coerce')

    return df_without_outliers

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

    # fit pipeline to  dataset + transform X_train
    X_train_preproc_ = preprocessing_pipeline.fit_transform(X_train, y_train)

    # Save trained preprocessing_pipeline
    with open('preprocessing_pipeline.pkl', 'wb') as file:
        pickle.dump(preprocessing_pipeline, file)

    # transform X_test
    X_test_preproc_ = preprocessing_pipeline.transform(X_test)

    # change in df with right column names
    X_train_preproc = pd.DataFrame(X_train_preproc_, columns = preprocessing_pipeline.get_feature_names_out(X_train.columns))
    X_test_preproc = pd.DataFrame(X_test_preproc_, columns = preprocessing_pipeline.get_feature_names_out(X_test.columns))
    y_train = pd.DataFrame(y_train, columns = ['price'])
    y_test = pd.DataFrame(y_test, columns = ['price'])

    # Concatenate test and train set
    X_all = pd.concat([X_train_preproc, X_test_preproc], axis=0, ignore_index=True)
    y_all = pd.concat([y_train, y_test], axis=0, ignore_index=True)

    # Concatenate X and y to have a full dataframe
    col = list(X_all.columns) + ['price']
    df_full = pd.concat([X_all, y_all], axis = 1, names = col)

    return X_train_preproc, X_test_preproc, y_train, y_test, X_all, y_all, df_full



def preprocess_input(input_data : pd.DataFrame, robust = True) -> pd.DataFrame:
    """ The preprocess_input function transforms the user input based on the pre-trained pipeline.
    """
    X_input = input_data

    file_path = '../../raw_data/preprocessing_pipeline.pkl'
    with open(file_path, 'rb') as file:
        trained_prepoc_pipeline = pickle.load(file)

    X_input_preproc = pd.DataFrame(trained_prepoc_pipeline.transform(X_input))

    return X_input_preproc
