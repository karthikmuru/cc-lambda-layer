import json
import urllib.parse
import boto3
import uuid
import cv2

s3_client = boto3.client('s3')

def face_recognition_handler(event, context):	

	# Get the object from the event and show its content type
	bucket = event['Records'][0]['s3']['bucket']['name']
	key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
	try:
			tmpkey = key.replace('/', '')
			download_path = '/tmp/{}'.format(tmpkey)
			
			s3_client.download_file(bucket, key, download_path)
			print("\n\ndownload_path:",download_path)
			
			video_cap = cv2.VideoCapture(download_path)
			video_cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
			_, frame = video_cap.read()

			print("Frame dims")
			print(len(frame))
			print(len(frame[0]))

			#print("CONTENT TYPE: " + response['ContentType'])
			# return response['ContentType']	
	except Exception as e:
			print(e)
			print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
			raise e