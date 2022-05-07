import base64
from io import BytesIO
import cv2
from PIL import Image
import numpy as np
from autocrop import Cropper

cropper = Cropper()

def base64_to_image(s) :
  img_data = base64.b64decode(s.encode('ascii'))
  npimg = np.fromstring(img_data, dtype = np.uint8)
  source = cv2.imdecode(npimg, 1)
  cropped_array = cropper.crop(source)

  try:
    return Image.fromarray(cropped_array).resize((160, 160))
  except Exception:
    return Image.fromarray(source).resize((160, 160))