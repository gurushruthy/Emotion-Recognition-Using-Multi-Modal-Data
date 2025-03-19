import h5py
import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K,optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input,Activation, Dense,Flatten,Dropout,MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D,Conv2D,concatenate,average
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.applications.mobilenet  import preprocess_input
from tensorflow.keras.utils import get_file,get_source_inputs
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint


def doubleGenerator(generator1,generator2):
  while True:
    for (x1,y1),(x2,y2) in zip(generator1,generator2):
      yield ([x1,x2],y1)
    

BATCH_SIZE=8
IMAGE_SIZE=224
EPOCHS=50
SEED=1

training_dir = "./data/train"

datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input
)


train_datagen1 = datagen.flow_from_directory(
    testing_dir,
    target_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE,
    shuffle = False,
    seed=SEED
    )

train_datagen2 = datagen.flow_from_directory(
    testing_dir,
    target_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE,
    shuffle = False,
    seed=SEED
    )


def compile_and_train(model, MODEL_NAME):
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=0.001),
                  metrics=['acc'])

    TRAINED_MODEL_PATH = './training_models/' + MODEL_NAME + '.{epoch:02d}--{val_acc:.2f}.hdf5'
    early_stop = EarlyStopping(monitor='val_loss', patience=7, mode='auto')
    checkpoint = ModelCheckpoint(TRAINED_MODEL_PATH, monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)

    history = model.fit_generator(
        dgenerator,
        steps_per_epoch=train_generator.samples / train_generator.batch_size,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[checkpoint, early_stop])

object_mg_model=load_model('./training_models/object_mg_model.hdf5')

for layer in object_mg_model.layers:
  layer._name = 'object_'+layer.name

object_output = object_mg_model.get_layer(name='object_dense_1').output

places_mg_model=load_model('./training_models/places_mg_model.hdf5')
for layer in places_mg_model.layers:
  layer._name = 'places_'+layer.name

places_output = places_mg_model.get_layer(name='places_dense_2').output
average_layer  = average([object_output, places_output])
late_fusion2_model = Model(inputs=[object_mg_model.input, places_mg_model.input], outputs=average_layer)
compile_and_train(late_fusion2_model,'late_fusion')

