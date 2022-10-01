 FROM python:3.8.12-buster
 COPY fidelifourche fidelifourche
 COPY mlops mlops
 COPY test_data test_data
 COPY requirements.txt requirements.txt
 RUN pip install -r requirements.txt
 CMD uvicorn fidelifourche.api.fast:app --host 0.0.0.0 --port $PORT
