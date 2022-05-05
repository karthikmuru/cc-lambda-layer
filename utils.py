import base64
from io import BytesIO
import cv2
from PIL import Image


def base64_to_image(s) :
  img_data = base64.b64decode(s.encode('ascii'))
  return Image.open(BytesIO(img_data))