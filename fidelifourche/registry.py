from fidelifourche.params import LOCAL_REGISTRY_PATH

import mlflow
from mlflow.tracking import MlflowClient

import glob
import os
import time
import pickle


def save_model(model,params,metrics):
    print("\nSave model to local disk...")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # save params
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH,"outputs", "params", timestamp + ".pickle")
        print(f"- params path: {params_path}")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH,"outputs","metrics", timestamp + ".pickle")
        print(f"- metrics path: {metrics_path}")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    # save model
    if model is not None:
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", timestamp)
        print(f"- model path: {model_path}")
        model.save(model_path)

    print("\n✅ data saved locally")

    return None

def load_local_model(save_copy_locally=False):

    print("\nLoad model from local disk...")

    # get latest model version
    model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")

    results = glob.glob(f"{model_directory}/*")
    if not results:
        return None

    model_path = sorted(results)[-1]
    print(f"- path: {model_path}")

    model = pickle.load_model(model_path)
    print("\n✅ model loaded from disk")

    return model


def get_model_version(stage="Production"):
    """
    Retrieve the version number of the latest model in the given stage
    - stages: "None", "Production", "Staging", "Archived"
    """

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

    mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

    client = MlflowClient()

    try:
        version = client.get_latest_versions(name=mlflow_model_name, stages=[stage])
    except:
        return None

    # check whether a version of the model exists in the given stage
    if not version:
        return None

    return int(version[0].version)
