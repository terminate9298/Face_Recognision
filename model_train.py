import numpy as np
import re,random,os,cv2,time,sys
import matplotlib.pyplot as plt

# from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Dropout , Flatten
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.05,
        height_shift_range=0.05,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

TRAIN_DIRECTORY = 'faces/'
TARGET_SIZE = (224,224)
INPUT_SHAPE = (224,224,3)
# NB_CLASSES = 4

train_generator = datagen.flow_from_directory(
    directory=TRAIN_DIRECTORY,
    target_size=TARGET_SIZE,
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

dictionary = train_generator.class_indices
dictionary = dict (zip(dictionary.values(),dictionary.keys()))
np.save('my_file.npy', dictionary) 

def baseline_model_vgg():

    input_1 = Input(shape = INPUT_SHAPE)
    base_model = VGGFace(model='vgg16' , include_top = False , input_shape =INPUT_SHAPE , pooling='avg')
    last_layer = base_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(100 , activation = 'relu')(x)
    x = Dropout(0.01)(x)
    out = Dense(NB_CLASSES, activation='softmax', name='classifier')(x)
    model = Model(base_model.input, out)

    model.compile(loss = 'categorical_crossentropy' , metrics = ['acc'] , optimizer = Adam(0.00001))
    model.summary()

    return model

def baseline_model_resnet():
    input_1 = Input(shape = INPUT_SHAPE)
    base_model = VGGFace(model='resnet50' , include_top = False , input_shape =INPUT_SHAPE , pooling='avg')
    last_layer = base_model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(100 , activation = 'relu')(x)
    x = Dropout(0.01)(x)
    out = Dense(NB_CLASSES, activation='softmax', name='classifier')(x)
    model = Model(base_model.input, out)

    model.compile(loss = 'categorical_crossentropy' , metrics = ['acc'] , optimizer = Adam(0.00001))
    model.summary()

    return model



if __name__ == "__main__":

	if sys.argv[1] == 'resnet':
		model = baseline_model_resnet()
	else:
		model = baseline_model_vgg()
	plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	history = model.fit_generator(
	      train_generator,
	      steps_per_epoch = train_generator.samples/train_generator.batch_size ,
	      epochs=20,
	      verbose=1)

	model.save('Model_VGGFace.h5')