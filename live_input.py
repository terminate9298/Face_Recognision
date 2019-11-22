import cv2
import os
import sys
import numpy as np
from random import randint
from keras.models import load_model
# Cascade Materials
cascade_file_path = 'Face_cascade.xml'
Face_cas = cv2.CascadeClassifier(cascade_file_path)

# Video Capture
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
if (cam.isOpened()== False): 
    print("Error opening video stream or file")
else:
    print('Reading Details')
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cam.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width ,height))


#Model Data
name_dict = np.load('my_file.npy',allow_pickle='TRUE').item()

model = load_model('Model_VGGFace.h5')
font = cv2.FONT_HERSHEY_SIMPLEX
def predict_class(img):
	img = cv2.resize(img,(224,224))
	img = np.reshape(img,[1,224,224,3])
	preds = model.predict(img)
	index = np.argmax(preds)
	return name_dict[index]+' -> '+str(preds[0][index]*100)+' %'


while True:
	ret, frame = cam.read()
	faces = Face_cas.detectMultiScale(cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY) , scaleFactor = 1.6, minNeighbors = 5 , minSize = (25,25) , flags = 0)
	for x,y,w,h in faces:
		cv2.rectangle(frame , (x,y) , (x+w , y+h) , (255,0,0) , 5)
		subimg = frame[y-10:y+h+10 , x-10 : x+w+10]
		class_name = predict_class(subimg)
		cv2.putText(frame, class_name, (x+w,y), font, .5, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.imshow("Face Recognision", frame)
	out.write(frame)
	if not ret:
		break
	k = cv2.waitKey(1)
	if k%256 == 27:
        # ESC pressed
		print("Escape hit, closing...")
		break
out.release()
cam.release()

cv2.destroyAllWindows()