# Face Recognision.

Face Recognision with VGG and Resnet model by using Transfer learning. 
### Requirements

* Keras
* Tensorflow
* Numpy 
* Pandas
* Opencv2
* Matplotlib
* Keras - VGG  [Documentation for Keras VGG](https://github.com/rcmalli/keras-vggface/)

### Files & Folders
* faces - Faces of persons kept inside each Folder named with person's Name.
* Images - Images of person Kept inside Directory Named with person's Name.
* video_frames - Frames of person kept inside folder with persons name if video is used to collect Images.
* face_extrator.py - To seperate faces from photos.
* frame_capture.py - Use to seperate frames from individual videos.
* live_input.py - To test final model after training which take input from Webcam and recognise the persons in video
* model_train.py - To Train Model after seperating each persons faces and keeping them in Correct Folders.

### Implementation	
* Input from video
	* Place the persons video in Main directory and start frame_capture.py file on each Video
	* Code - 
		'''Shell
			python frame_capture.py name_1.mp4
		'''
	* This will place frames from each video in video_frames folders inside name_1 folder.
* Faces from Frames
	* You can take input from Video or collect individual photos of each person and keep inside images folder
	* Code - 
		'''Shell
		   python face_extractor.py video_frames/name_1
		   or 
		   python face_extractor.py images/name_1
		'''
	* This will extract faces from photos and place them faces folder inside there corresponding Folders.

* Start Training -
	* Before starting Make sure the Directories are formed in correct order.
	* faces Folder have structure 
	-> faces
	----> name_1
	--------> file_of_name_1
	--------> file_of_name_1
	--------> file_of_name_1
	--------> file_of_name_1
	----> name_2
	--------> file_of_name_2
	--------> file_of_name_2
	--------> file_of_name_2
	--------> file_of_name_2
	----> name_3
	--------> file_of_name_3
	--------> file_of_name_3
	--------> file_of_name_3
	--------> file_of_name_3
	...

	* Start the training with Command " python model_train.py resnet " or " python model_train.py vgg " to train model with either Resnet50 or VGG16 Model . VGG16 is smaller model and trains faster and requires less RAM Compare to Resnet50 model.
	* If everything is successful you will find named Model_VGGFace.h5 inside main directory.

* Test Model - 
	* Start the video input from Webcam by command " python live_input.py "
	* If everything is Successful you will find Rectangle around faces with name of person written over it.

### Note 
* You can change values inside file face_extractor.py file and live_input.py file and play with scaleFactor , minNeighbors of HaarCascade .These values work fine for me.
* Some Manual Assist may require after face_extration as some wrong files may come along with faces.


### Related Files
* https://github.com/rcmalli/keras-vggface