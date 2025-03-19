import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import string
import os
import glob
from PIL import Image
from time import time

from keras import Input, layers
from keras import optimizers
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.utils import to_categorical

!unzip '/content/drive/MyDrive/archive (1).zip' -d '/content/drive/MyDrive/glove6b'

token_path = "/content/drive/MyDrive/flickr_text/Flickr8k.token.txt"
train_images_path = '/content/drive/MyDrive/flickr_text/Flickr_8k.trainImages.txt'
test_images_path = '/content/drive/MyDrive/flickr_text/Flickr_8k.testImages.txt'
images_path = '/content/drive/MyDrive/flickr_dataset/Images/'
glove_path = '/content/drive/MyDrive/glove6b'

doc = open(token_path,'r').read()
print(doc[:410])

descriptions = dict()
for line in doc.split('\n'):
        tokens = line.split()
        if len(line) > 2:
          image_id = tokens[0].split('.')[0]
          image_desc = ' '.join(tokens[1:])
          if image_id not in descriptions:
              descriptions[image_id] = list()
          descriptions[image_id].append(image_desc)

table = str.maketrans('', '', string.punctuation)
for key, desc_list in descriptions.items():
    for i in range(len(desc_list)):
        desc = desc_list[i]
        desc = desc.split()
        desc = [word.lower() for word in desc]
        desc = [w.translate(table) for w in desc]
        desc_list[i] =  ' '.join(desc)



pic = '280706862_14c30d734a.jpg'
x=plt.imread(images_path+pic)
plt.imshow(x)
plt.show()
descriptions['280706862_14c30d734a'][0]

vocabulary = set()
for key in descriptions.keys():
        [vocabulary.update(d.split()) for d in descriptions[key]]
print('Original Vocabulary Size: %d' % len(vocabulary))

lines = list()
for key, desc_list in descriptions.items():
    for desc in desc_list:
        lines.append(key + ' ' + desc)
new_descriptions = '\n'.join(lines)

doc = open(train_images_path,'r').read()
dataset = list()
for line in doc.split('\n'):
    if len(line) > 1:
      identifier = line.split('.')[0]
      dataset.append(identifier)

train = set(dataset)

img = glob.glob(images_path + '*.jpg')
train_images = set(open(train_images_path, 'r').read().strip().split('\n'))
train_img = []
for i in img: 
    if i[len(images_path):]    in train_images:
        train_img.append(i)

test_images = set(open(test_images_path, 'r').read().strip().split('\n'))
test_img = []
for i in img: 
    if i[len(images_path):] in test_images: 
        test_img.append(i)

train_descriptions = dict()
for line in new_descriptions.split('\n'):
    tokens = line.split()
    image_id, image_desc = tokens[0], tokens[1:]
    if image_id in train:
        if image_id not in train_descriptions:
            train_descriptions[image_id] = list()
        desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
        train_descriptions[image_id].append(desc)

all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)

word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

print('Vocabulary = %d' % (len(vocab)))

ixtoword = {}
wordtoix = {}
ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

vocab_size = len(ixtoword) + 1

all_desc = list()
for key in train_descriptions.keys():
    [all_desc.append(d) for d in train_descriptions[key]]
lines = all_desc
max_length = max(len(d.split()) for d in lines)

print('Description Length: %d' % max_length)

embeddings_index = {} 
f = open(os.path.join(glove_path, 'glove.6B.200d.txt'), encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
#print(embeddings_index)

embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in wordtoix.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#import numpy as np
#from numpy import array
#from keras.preprocessing import sequence
#from keras.preprocessing import image
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
model = InceptionV3(weights='imagenet')

model_new = Model(model.input, model.layers[-2].output)

#img_path=
def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess(image) 
    fea_vec = model_new.predict(image) 
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

encoding_test = {}
for img in test_img:
    encoding_test[img[len(images_path):]] = encode(img)

print(encoding_test)


import os

data = str(encoding_test)
with open('/content/test.txt', 'w') as writefile:
    writefile.write(data)

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.summary()

model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam')

def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key+'.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)

            if n==num_photos_per_batch:
                yield ([array(X1), array(X2)], array(y))
                X1, X2, y = list(), list(), list()
                n=0

from keras.models import load_model
model = load_model('/content/drive/MyDrive/models/model_15.h5')

epochs = 30
batch_size = 3
steps = len(train_descriptions)//batch_size

#generator = data_generator(train_descriptions, train_features, wordtoix, max_length, batch_size)
#model.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1)
os.mkdir("models")
for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, wordtoix, max_length, batch_size)
    model.fit_generator(generator, epochs=1, steps_per_epoch= steps, verbose=1)
    model.save("models/model_" + str(i) + ".h5")

def beam_search_predictions(image, beam_index = 3):
    start = [wordtoix["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = model.predict([image,par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]
    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption

#img_path='/content/'
#pic = '468911753_cc595f5da0.jpg'
pic='1000268201_693b08cb0e.jpg'
#image = encoding_test[pic].reshape((1,2048))
image = encoding[pic].reshape((1,2048))
x=plt.imread(img_path+pic)
#x=plt.imread(pic)
plt.imshow(x)
plt.show()

print("Beam Search, K = 3:",beam_search_predictions(image, beam_index = 3))

#pic = list(encoding_test.keys())[80]
pic='3767841911_6678052eb6.jpg'
image = encoding_test[pic].reshape((1,2048))
x=plt.imread(images_path+pic)
plt.imshow(x)
plt.show()
ch='.jpg'
pic=pic.replace(ch,'')
print("Predicted_caption:",beam_search_predictions(image, beam_index = 3))
print("Expected Output:",descriptions[pic][1])

import cPickle as pickle
import caption_generator
import numpy as np
from keras.preprocessing import sequence
import nltk

cg = caption_generator.CaptionGenerator()

def process_caption(caption):
	caption_split = caption.split()
	processed_caption = caption_split[1:]
	try:
		end_index = processed_caption.index('<end>')
		processed_caption = processed_caption[:end_index]
	except:
		pass
	return " ".join([word for word in processed_caption])

def get_best_caption(captions):
    captions.sort(key = lambda l:l[1])
    best_caption = captions[-1][0]
    return " ".join([cg.index_word[index] for index in best_caption])

def get_all_captions(captions):
    final_captions = []
    captions.sort(key = lambda l:l[1])
    for caption in captions:
        text_caption = " ".join([cg.index_word[index] for index in caption[0]])
        final_captions.append([text_caption, caption[1]])
    return final_captions

def generate_captions(model, image, beam_size):
	start = [cg.word_index['<start>']]
	captions = [[start,0.0]]
	while(len(captions[0][0]) < cg.max_cap_len):
		temp_captions = []
		for caption in captions:
			partial_caption = sequence.pad_sequences([caption[0]], maxlen=cg.max_cap_len, padding='post')
			next_words_pred = model.predict([np.asarray([image]), np.asarray(partial_caption)])[0]
			next_words = np.argsort(next_words_pred)[-beam_size:]
			for word in next_words:
				new_partial_caption, new_partial_caption_prob = caption[0][:], caption[1]
				new_partial_caption.append(word)
				new_partial_caption_prob+=next_words_pred[word]
				temp_captions.append([new_partial_caption,new_partial_caption_prob])
		captions = temp_captions
		captions.sort(key = lambda l:l[1])
		captions = captions[-beam_size:]

	return captions

def test_model(weight, img_name, beam_size = 3):
	encoded_images = pickle.load( open( "encoded_images.p", "rb" ) )
	model = cg.create_model(ret_model = True)
	model.load_weights(weight)

	image = encoded_images[img_name]
	captions = generate_captions(model, image, beam_size)
	return process_caption(get_best_caption(captions))
	#return [process_caption(caption[0]) for caption in get_all_captions(captions)] 

def bleu_score(hypotheses, references):
	return nltk.translate.bleu_score.corpus_bleu(references, hypotheses)

def test_model_on_images(weight, img_dir, beam_size = 3):
	imgs = []
	captions = {}
	with open(img_dir, 'rb') as f_images:
		imgs = f_images.read().strip().split('\n')
	encoded_images = pickle.load( open( "encoded_images.p", "rb" ) )
	model = cg.create_model(ret_model = True)
	model.load_weights(weight)

	f_pred_caption = open('predicted_captions.txt', 'wb')

	for count, img_name in enumerate(imgs):
		#print "Predicting for image: "+str(count)
		image = encoded_images[img_name]
		image_captions = generate_captions(model, image, beam_size)
		best_caption = process_caption(get_best_caption(image_captions))
		captions[img_name] = best_caption
		print img_name+" : "+str(best_caption)
		f_pred_caption.write(img_name+"\t"+str(best_caption))
		f_pred_caption.flush()
	f_pred_caption.close()

	f_captions = open('/content/drive/MyDrive/flickr_text/Flickr8k.token.txt', 'rb')
	captions_text = f_captions.read().strip().split('\n')
	image_captions_pair = {}
	for row in captions_text:
		row = row.split("\t")
		row[0] = row[0][:len(row[0])-2]
		try:
			image_captions_pair[row[0]].append(row[1])
		except:
			image_captions_pair[row[0]] = [row[1]]
	f_captions.close()
	
	hypotheses=[]
	references = []
	for img_name in imgs:
		hypothesis = captions[img_name]
		reference = image_captions_pair[img_name]
		hypotheses.append(hypothesis)
		references.append(reference)

	return bleu_score(hypotheses, references)

if __name__ == '__main__':
	weight = '/content/drive/MyDrive/models/model_15.h5'
	test_image = '3155451946_c0862c70cb.jpg'
	test_img_dir = '/content/drive/MyDrive/flickr_text/Flickr_8k.testImages.txt'
	#print test_model(weight, test_image)
	print test_model_on_images(weight, test_img_dir, beam_size=3)