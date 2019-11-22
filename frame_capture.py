import cv2
import os
import sys
import numpy as np
from random import randint

def frame_extractor(path , dir_name):
    vidObj = cv2.VideoCapture(path) 
    count = 0
    while 1: 
        success, image = vidObj.read() 
        # image = np.rot90(image)
        if not success:
            break
        cv2.imwrite("video_frames/%s/frame%d.jpg" % (dir_name , count), image) 
        count += 1
        cv2.destroyAllWindows()

def make_directory(dir):
    if not os.path.exists('video_frames'):
        os.makedirs('video_frames')
    if not os.path.exists('video_frames/'+dir):
        os.makedirs('video_frames/'+dir)
    print('Directory Build...')
    print('Starting to Save Images ...')

if __name__ == "__main__":
    
    dir_name = str(sys.argv[1])
    print(dir_name)
    make_directory(dir_name.split('.')[0])
    frame_extractor(sys.argv[1] , dir_name.split('.')[0])
