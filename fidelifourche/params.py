"""
taxifare model package params
load and validate the environment variables in the `.env`
"""

import os

#LOCAL_DATA_PATH = os.path.join('..','test_data')
#LOCAL_REGISTRY_PATH = os.path.join('..','mlops')

LOCAL_DATA_PATH = os.path.join(os.getenv('HOME'),"code","edithsarafis", "fidelifourche","raw_data")
LOCAL_REGISTRY_PATH = os.path.join(os.getenv('HOME'),"code","edithsarafis", "fidelifourche",'mlops')


DTYPES_RAW = {
    "customer_id": "O",
    "zip": "O",
    "bool_churn": "int8",
    "aov": "float32",
    "weight": "float32",
    "delivery_type": "O",
    "carrier": "O",
    "created_at": "O",
    "order_id": "O",
    "share_refunds": "float32",
    "share_discount": "float32",
    "Baby": "float32",
    "Vrac": "float32",
    "Sale": "float32",
    "Bois": "float32",
    "Mais": "float32",
    "Sucr": "float32",
    "Sant": "float32",
    "Alco": "float32",
    "Beau": "float32",
    "acquisition_channel": "O",
    "delay": "float32",
    "raw_subject": "O",
    "ticket_at": "O",
    "value": "O"
}
