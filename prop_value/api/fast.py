import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
#import required functions from ml_logic
#from taxifare.ml_logic.registry import load_model


app = FastAPI()
#model = load_model()#TODO load the model function from ml_logic
#assert model is not None, "Model is not loaded"
#app.state.model = model

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
        property_type: str, #Appartement or Maison
        built: str, #Vente or Vente en l’état futur d’achèvement
        number_of_rooms: float #3.0
    ):
    """
    Make a single price prediction for the property.
    """
    # Create X_pred DataFrame
    X_pred = pd.DataFrame(locals(), index=[0])
    # Preprocess features
    #X_processed = preprocess_features(X_pred) #TODO: create preprocess_features function

    #call pre-loaded model to get prediction
    y_pred = model.predict(X_processed) #TODO: create predict function in ml_logic
    y_pred = round(float(y_pred), 2)
    y_pred = 10000
    return {'predicted_price': y_pred}
