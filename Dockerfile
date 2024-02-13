FROM python:3.8.12-buster

WORKDIR /prod

COPY model_file model_file
COPY raw_data raw_data

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY prop_value prop_value
COPY setup.py setup.py
RUN pip install .

COPY Makefile Makefile

CMD uvicorn prop_value.api.fast:app --host 0.0.0.0 --port $PORT
