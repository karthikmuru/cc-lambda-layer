import os
import numpy as np
import json
import urllib.parse
import boto3
import cv2
import time
import random
import uuid

import torch
import torchvision.transforms as transforms
from PIL import Image

import build_custom_model
import utils


LABELS_DIR = os.path.abspath('./checkpoint/labels.json')
MODEL_PATH = os.path.abspath("./checkpoint/model_vggface2_best.pth")
student_table_name = 'student_table'

print('Loading function..')

s3_client = boto3.client('s3')    
dynamo_db_client = boto3.client('dynamodb', region_name='us-east-1')

start = time.time()
with open(LABELS_DIR, encoding="utf8") as f:
    LABELS = json.load(f)
print(f"labels: {LABELS}")

print('Loading model weights...')

model = build_custom_model.build_model(len(LABELS)).to(torch.device('cpu'))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))['model'])
model.eval()
end = time.time()

print("Total time taken to load model : " , end - start)

def handler(event, context):
    print('FACE-RECOGNIZER-PRO v5')
    
    try:
        data = json.loads(event['body'])    
        img = utils.base64_to_image(data['image'])
        print(img)
        prediction = predict(img)
        print("Prediction generated!")
        print(prediction)
            
        data = get_student_data(prediction)        

        return {'prediction': data}

    except Exception as e:
        print("EXCEPTION!")
        print(e)
        raise e
    
# Make prediction using the model, given the image.
def predict(img):

    img_tensor = transforms.ToTensor()(img).unsqueeze_(0).to(torch.device('cpu'))
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs.data, 1)
    result = LABELS[np.array(predicted.cpu())[0]]

    return result

# Fetch student data from DynamoDB
def get_student_data(name) :
    response = dynamo_db_client.get_item(
    TableName = student_table_name,
        Key = {
            'name': {'S' : name}
        }
    )
    
    data = {
        'student_year': response['Item']['year']['S'],
        'student_major': response['Item']['major']['S'],
        'student_name': response['Item']['name']['S'],
    }

    return data