import pandas as pd
import numpy as np
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input

## Surpress warning


uploaded_file = ('bg2.jpg')

## Load pretrained model
emotion_model = tf.keras.models.load_model('late_fusion2_model.h5')

## After image uploaded
if uploaded_file is not None:
    ## Image preprocessing
    image = Image.open(uploaded_file)
    size = (224, 224)
    im = ImageOps.fit(image, size, Image.ANTIALIAS)
    im = img_to_array(im)
    im = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
    im = preprocess_input(im)

    ## Predictions
    preds = emotion_model.predict([im, im])
    predicted_index = np.argmax(preds, axis=1)[0]
    labels = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
    predicted_classes = str(labels[predicted_index])

    ## Display

    print('Predicted Emotion: ' + predicted_classes)
    print('Emotion Distribution:')
    print(pd.DataFrame({'emotion_classes': labels\
                           , 'predicted_probability': preds[0]}))
