import pandas as pd
#from google.cloud import bigquery
import numpy as np
import matplotlib as plt
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

def download_all_csv():
    ## list all the number of regions we can download
    regions = list(np.arange(1,96))
    regions = list(np.delete(regions, [20-1, 57-1, 67-1, 68-1]))
    regions = regions + ['2A', '2B']

    #download all csv file
    for index, num in enumerate(regions):
        if index < 9 :
            url = f'https://dvf-api.data.gouv.fr/dvf/csv/?dep=0{num}'
        else :
            url = f'https://dvf-api.data.gouv.fr/dvf/csv/?dep={num}'

        path = Path(f'../raw_data/dvf_{num}.csv')

        #checking if the file already is in raw_data
        if not path.is_file():
            df = pd.read_csv(url)
            df.to_csv(path)
            print(f'Downloaded region n° {num} 💪')
        else:
            print(f'region n° {num} is already downloaded ! 🚀')



def clean_data(df_dvf: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - removing irrelevant columns
    - choose relevant values inside columns
    - assigning correct dtypes to each column
    - remove NaN and dupliactes
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
                    'Vente en l’état futur d’achèvement' : 'off-plan'}
    trans_dict_type = {'Appartement' : 'appartment',
                   'Maison' : 'house'}
    df_useful= df_useful.replace({'built' : trans_dict_built,
                             'property type' : trans_dict_type})


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

#get data
def get_data():
    df_dvf = pd.read_csv('../raw_data/dvf_93.csv')

    return df_dvf
