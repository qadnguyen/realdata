import pandas as pd
import numpy as np


def get_citycode(postal_code: int) -> int:
    """
    Get the city code based on the postal code.
    """
    #load pickle file with city codes and postal codes
    file_path = '../../raw_data/codes_dict.pkl'
    with open(file_path, 'rb') as file:
        citycodes = pickle.load(file)
        assert citycodes is not None, "Citycodes are not loaded"
    city_code = citycodes[citycodes['postal_code'] == postal_code]['city_code'].values[0]
    return city_code
