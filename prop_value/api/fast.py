import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
from xgboost import XGBRegressor

#import required functions from ml_logic
from prop_value.ml_logic.preprocessor import preprocess_input
from prop_value.ml_logic.geocoding import get_citycode


app = FastAPI()
#load the model via pickle file
path_to_file = '../../model_file/xgb_model.pkl'
with open(path_to_file, 'rb') as file:
    model = pickle.load(file)
assert model is not None, "Model is not loaded"
app.state.model = model

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#new prediciton endpoint
# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict_price")
def predict_price(
        living_area: float,  # 110.0
        latitude: float,    # -73.950655
        longitude: float,     # 40.783282
        property_type: str, #appartment or house
        built: str, #built or off-plan
        number_of_rooms: float, #3.0
        postal_code: int, # 1000 #TODO: calculate based on user address input
    ):
    """
    Make a single price prediction for the property.
    """
    # Create X_pred DataFrame
    X_pred = pd.DataFrame(locals(), index=[0])
    # City code needs to be retrieved from dictionary via postal code
    X_pred['city'] = get_citycode(X_pred['postal_code'])
    # Preprocess features
    X_processed = preprocess_input(X_pred)

    #call pre-loaded model to get prediction
    y_pred = model.predict(X_processed)
    print('Predicted price: ' + str(y_pred))
    y_pred = round(float(y_pred), 2)
    return {'predicted_price': y_pred}
