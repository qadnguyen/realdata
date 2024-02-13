import pandas as pd
import numpy as np
import pickle
import os


def get_citycode(postal_code: float) -> float:
    """
    Get the city code based on the postal code.
    """
    #load pickle file with city codes and postal codes
    file_path = os.path.abspath('raw_data/codes_dict.pkl')

    with open(file_path, 'rb') as file:
        citycodes = pickle.load(file)
        assert citycodes is not None, "Citycodes are not loaded"
    #get city code
    postal_code = postal_code.iloc[0]
    city_code = citycodes[postal_code]
    return city_code
