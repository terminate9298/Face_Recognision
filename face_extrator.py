import cv2
import os
import sys
from random import randint

# Cascade File 
cascade_file_path = 'Face_cascade.xml'
Face_cas = cv2.CascadeClassifier(cascade_file_path)

def face_function(path):
    image = cv2.imread(path)
    image_gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    # If the Faces of person are not Detecting perfectly try Changing 
    # ScaleFactor from 1.05 to 2.0
    # Minneighbours from 3 to 6
    FACES = Face_cas.detectMultiScale(image_gray , scaleFactor = 2.0, minNeighbors = 5 , minSize = (25,25) , flags = 0)
    
    for x,y,w,h in FACES:
        subimg = image[y-10:y+h+10 , x-10 : x+w+10]
        cv2.imwrite('faces/'+str(path.split('/')[-2])+'/'+str(path.split('/')[-1].split('.')[0])+'_'+str(randint(0,10))+'.jpg' , subimg)


def make_directory(dir):
    if not os.path.exists('faces'):
        os.makedirs('faces')
    if not os.path.exists('faces/'+dir):
        os.makedirs('faces/'+dir)

if __name__ == "__main__":

    # Directory to read
    dir_name = str(sys.argv[1])
    dir_list = dir_name.split('/')
    make_directory(dir_list[1])

    for files in os.listdir(dir_name):
        print(files)
        face_function(dir_name+'/'+files)
