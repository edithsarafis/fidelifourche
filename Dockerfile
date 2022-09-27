# use case 1: apple silicon - build for local use
#
#   docker build -f Dockerfile_silicon --tag=$IMAGE .
#   docker run -it -e PORT=8000 -p 8000:8000 --env-file .env $IMAGE

# use case 2: apple silicon: build for intel prod server
#
# PROD_IMAGE=prod-$IMAGE
# docker build -t $MULTI_REGION/$PROJECT/$PROD_IMAGE --platform linux/amd64 .
# docker push $MULTI_REGION/$PROJECT/$PROD_IMAGE
# gcloud run deploy \
      --image $MULTI_REGION/$PROJECT/$PROD_IMAGE \
      --region $REGION \
      --memory $MEMORY \
      --env-vars-file .env.yaml


FROM python:3.8.12-bullseye
COPY taxifare /taxifare
COPY requirements_silicon.txt /requirements.txt
RUN pip install tensorflow-aarch64 -f https://tf.kmtea.eu/whl/stable.html
RUN pip install -r requirements.txt
CMD uvicorn taxifare.api.fast:app --host 0.0.0.0 --port $PORT
