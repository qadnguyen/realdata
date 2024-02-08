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
