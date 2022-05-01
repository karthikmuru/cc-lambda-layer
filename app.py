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


LABELS_DIR = os.path.abspath('./checkpoint/labels.json')
MODEL_PATH = os.path.abspath("./checkpoint/model_vggface2_best.pth")

student_table_name = 'student_table'
RESULT_QUEUE_NAME = 'result_queue.fifo'
AWS_QUEUE_URL = 'QueueUrl'
#message group id of result fifo queue
MESSAGE_GROUP_ID = 'RESPONSE_GROUP'

print('Loading function..')

s3_client = boto3.client('s3')    
dynamo_db_client = boto3.client('dynamodb')
sqs_client = boto3.client('sqs')

start = time.time()
with open(LABELS_DIR, encoding="utf8") as f:
    LABELS = json.load(f)
print(f"labels: {LABELS}")

print('Loading model weights...')

model = build_custom_model.build_model(len(LABELS)).to(torch.device('cpu'))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))['model'])
end = time.time()

print("Total time taken to load model : " , end - start)

def handler(event, context):
    print('my lambda')


    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    
    try:
        #response = s3.get_object(Bucket=bucket, Key=key)
        tmpkey = key.replace('/', '')
        download_path = '/tmp/{}'.format(tmpkey)
        
        s3_client.download_file(bucket, key, download_path)
        print("\n\ndownload_path:",download_path)
        
        video_cap = cv2.VideoCapture(download_path)

        ret, frame = video_cap.read()
        print("EXTRACTED FRAME!")
        prediction = predict(frame)
        print("Prediction generated!")
        print(prediction)
            
        data = get_student_data(prediction)

        print(data)
        
        send_sqs_message(data)

        return {'prediction': data}

    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e
    
    return {'message': 'Error in execution!'}

def predict(image_array):

    img = Image.fromarray(image_array)

    img_tensor = transforms.ToTensor()(img).unsqueeze_(0).to(torch.device('cpu'))
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs.data, 1)
    result = LABELS[np.array(predicted.cpu())[0]]

    return result

def get_student_data(name) :
    response = dynamo_db_client.get_item(
    TableName=student_table_name,
        Key={
            'name': {'S' : name}
        }
    )
    
    data = {
        'student_year': response['Item']['major']['S'],
        'student_major': response['Item']['year']['S'],
        'student_name': response['Item']['name']['S'],
    }

    return data

def send_sqs_message(sqs_message) :
    sqs_message['video_interval_id'] = str(random.randrange(1, 300, 1)) # TODO: change to id from video/image name
    queue_url = get_response_queue_url(sqs_client)
    sqs_client.send_message(QueueUrl=queue_url,MessageBody=str(sqs_message),MessageGroupId=MESSAGE_GROUP_ID,MessageDeduplicationId = str(uuid.uuid4()) )


def get_response_queue_url(sqs_client):
    queue_url = None
    try:
        queue_url = sqs_client.get_queue_url(QueueName=RESULT_QUEUE_NAME)[AWS_QUEUE_URL]
    except sqs_client.exceptions.QueueDoesNotExist:
        sqs_client.create_queue(QueueName=RESULT_QUEUE_NAME)
        queue_url = sqs_client.get_queue_url(QueueName=RESULT_QUEUE_NAME)[AWS_QUEUE_URL]
    return queue_url


# if __name__ == "__main__":
#     path = '/Users/karthik/Desktop/ASU/CC/video3.h264'
#     print(LABELS)
#     print(path)
#     video_cap = cv2.VideoCapture(path)
#     ret, frame = video_cap.read()