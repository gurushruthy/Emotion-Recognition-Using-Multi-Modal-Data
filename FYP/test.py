import tweepy
import emoji
import re
from nltk.parse import CoreNLPDependencyParser
from nltk.stem import WordNetLemmatizer
import skfuzzy as fuzz
from nrclex import NRCLex
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from bs4 import BeautifulSoup
from autocorrect import Speller
import string
from keras.preprocessing import sequence
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.models import load_model
import math
from collections import Counter
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
from urllib import request as requi
from io import BytesIO
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.utils import to_categorical
from keras import Input, layers
from keras import optimizers
from keras.optimizers import Adam
import os
import glob
from time import time


Lemmatizer = WordNetLemmatizer()
check = Speller(lang='en')

consumer_key = 'keHJwXehkR7bpkPhDUzleub3O'
consumer_secret = 'qFgTatBCZYnl5k85HeLteP9wS0LE0ty4sIHUb6eS1xff47n9AG'
access_token = '1364272365719281668-l1l1dtt8WlqizuPwnyBG1FixKik5Co'
access_token_secret = 'f8RrLfOHoKQLXKDlQV4C1ORYnWYU8BPjBWRu3kqcu1P3x'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

app = Flask(__name__)

x_s = np.arange(0, 1.1, 0.1)
x_h = np.arange(0, 1.1, 0.1)
x_a = np.arange(0, 1.1, 0.1)
x_f = np.arange(0, 1.1, 0.1)
x_op = np.arange(0, 10, 1)
print(x_s)
print(x_h)
print(x_a)
print(x_f)
print(x_op)

h_lo = fuzz.trimf(x_h, [0, 0, 0.4])
h_md = fuzz.trimf(x_h, [0, 0.6, 1])
h_hi = fuzz.trimf(x_h, [0.6, 1, 1])
s_lo = fuzz.trimf(x_s, [0, 0, 0.4])
s_md = fuzz.trimf(x_s, [0, 0.6, 1])
s_hi = fuzz.trimf(x_s, [0.6, 1, 1])
a_lo = fuzz.trimf(x_a, [0, 0, 0.4])
a_md = fuzz.trimf(x_a, [0, 0.6, 1])
a_hi = fuzz.trimf(x_a, [0.6, 1, 1])
f_lo = fuzz.trimf(x_f, [0, 0, 0.4])
f_md = fuzz.trimf(x_f, [0, 0.6, 1])
f_hi = fuzz.trimf(x_f, [0.6, 1.0, 1.0])
op_hap = fuzz.trimf(x_op, [0, 0, 3])
op_sad = fuzz.trimf(x_op, [0, 3, 6])
op_fea = fuzz.trimf(x_op, [3, 6, 9])
op_ang = fuzz.trimf(x_op, [6, 9, 10])
fig, (ax0, ax1, ax3, ax4, ax2) = plt.subplots(nrows=5, figsize=(8, 9))
ax0.plot(x_h, h_lo, 'b', linewidth=1.5, label='Low')
ax0.plot(x_h, h_md, 'g', linewidth=1.5, label='Medium')
ax0.plot(x_h, h_hi, 'r', linewidth=1.5, label='High')
ax0.set_title('happy')
ax0.legend()

ax1.plot(x_s, s_lo, 'b', linewidth=1.5, label='Low')
ax1.plot(x_s, s_md, 'g', linewidth=1.5, label='Medium')
ax1.plot(x_s, s_hi, 'r', linewidth=1.5, label='High')
ax1.set_title('sad')
ax1.legend()

ax3.plot(x_a, a_lo, 'b', linewidth=1.5, label='Low')
ax3.plot(x_a, a_md, 'g', linewidth=1.5, label='Medium')
ax3.plot(x_a, a_hi, 'r', linewidth=1.5, label='High')
ax3.set_title('angry')
ax3.legend()

ax4.plot(x_f, f_lo, 'b', linewidth=1.5, label='Low')
ax4.plot(x_f, f_md, 'g', linewidth=1.5, label='Medium')
ax4.plot(x_f, f_hi, 'r', linewidth=1.5, label='High')
ax4.set_title('fear')
ax4.legend()

ax2.plot(x_op, op_hap, 'b', linewidth=1.5, label='happy')
ax2.plot(x_op, op_sad, 'y', linewidth=1.5, label='sad')
ax2.plot(x_op, op_ang, 'g', linewidth=1.5, label='anger')
ax2.plot(x_op, op_fea, 'r', linewidth=1.5, label='fear')
ax2.set_title('Output')
ax2.legend()

#Intensity

x_m = np.arange(0, 1.1, 0.1)
x_ar = np.arange(0, 1.1, 0.1)
x_op_ei = np.arange(0, 9, 1)
m_lo = fuzz.trimf(x_m, [0, 0, 0.4])
m_md = fuzz.trimf(x_m, [0, 0.6, 1])
m_hi = fuzz.trimf(x_m, [0.6, 1, 1])
ar_lo = fuzz.trapmf(x_ar, [0, 0.025, 0.475, 0.5])
ar_md = fuzz.trimf(x_ar, [0.475, 0.5, 0.525])
ar_hi = fuzz.trapmf(x_ar, [0.5, 0.525, 0.975, 1])
op_1 = fuzz.trimf(x_op_ei, [0, 0, 2])
op_2 = fuzz.trimf(x_op_ei, [0, 2, 4])
op_3 = fuzz.trimf(x_op_ei, [2, 4, 6])
op_4 = fuzz.trimf(x_op_ei, [4, 6, 8])
op_5 = fuzz.trimf(x_op_ei, [6, 8, 8])

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_m, m_lo, 'b', linewidth=1.5, label='Low')
ax0.plot(x_m, m_md, 'g', linewidth=1.5, label='Medium')
ax0.plot(x_m, m_hi, 'r', linewidth=1.5, label='High')
ax0.set_title('emotion_assossciation')
ax0.legend()

ax1.plot(x_ar, ar_lo, 'b', linewidth=1.5, label='Low')
ax1.plot(x_ar, ar_md, 'g', linewidth=1.5, label='Medium')
ax1.plot(x_ar, ar_hi, 'r', linewidth=1.5, label='High')
ax1.set_title('arousal')
ax1.legend()

ax2.plot(x_op_ei, op_1, 'b', linewidth=1.5, label='ext_low')
ax2.plot(x_op_ei, op_2, 'y', linewidth=1.5, label='low')
ax2.plot(x_op_ei, op_3, 'g', linewidth=1.5, label='moderate')
ax2.plot(x_op_ei, op_4, 'r', linewidth=1.5, label='high')
ax2.plot(x_op_ei, op_5, 'r', linewidth=1.5, label='ext_high')
ax2.set_title('Output')
ax2.legend()

negators = ["not", "no", "none", "nothing", "never", "nobody", "nowhere", "neither", "nor", "hardly", "barely","scarcely", "falsely"]
negator_adverbs = ["hardly", "barely", "scarcely", "falsely"]
pos_conj = ["and"]
prons = ["everyone", "one", "once", "thing", "all", "ever"]
#
d={'outraged': ['0.964', 'anger'], 'brutality': ['0.959', 'anger'], 'hatred': ['0.953', 'anger'], 'hateful': ['0.940', 'anger'], 'terrorize': ['0.939', 'anger'], 'infuriated': ['0.938', 'anger'], 'violently': ['0.938', 'anger'], 'furious': ['0.929', 'anger'], 'enraged': ['0.927', 'anger'], 'furiously': ['0.927', 'anger'], 'screwyou': ['0.924', 'anger'], 'murderer': ['0.922', 'anger'], 'fury': ['0.922', 'anger'], 'execution': ['0.917', 'anger'], 'angered': ['0.916', 'anger'], 'savagery': ['0.915', 'anger'], 'slaughtering': ['0.914', 'anger'], 'veryangry': ['0.913', 'anger'], 'assassinate': ['0.912', 'anger'], 'annihilation': ['0.912', 'anger'], 'fuckoff': ['0.912', 'anger'], 'rage': ['0.911', 'anger'], 'loathe': ['0.909', 'anger'], 'damnation': ['0.906', 'anger'], 'fucktard': ['0.906', 'anger'], 'homicidal': ['0.906', 'anger'], 'roadrage': ['0.906', 'anger'], 'furor': ['0.900', 'anger'], 'hostile': ['0.898', 'anger'], 'annihilate': ['0.898', 'anger'], 'murder': ['0.897', 'anger'], 'raging': ['0.896', 'anger'], 'explosive': ['0.894', 'anger'], 'infuriates': ['0.894', 'anger'], 'pissed': ['0.894', 'anger'], 'ferocious': ['0.894', 'anger'], 'obliterated': ['0.894', 'anger'], 'rape': ['0.894', 'anger'], 'vengeful': ['0.894', 'anger'], 'sopissed': ['0.894', 'anger'], 'killing': ['0.893', 'anger'], 'combative': ['0.891', 'anger'], 'gofuckyourself': ['0.886', 'anger'], 'vengeance': ['0.886', 'anger'], 'wrath': ['0.885', 'anger'], 'torment': ['0.885', 'anger'], 'vicious': ['0.884', 'anger'], 'massacre': ['0.882', 'anger'], 'threatening': ['0.882', 'anger'], 'abhorrent': ['0.875', 'anger'], 'pissoff': ['0.875', 'anger'], 'bloodthirsty': ['0.875', 'anger'], 'fighting': ['0.868', 'anger'], 'attacking': ['0.865', 'anger'], 'annihilated': ['0.865', 'anger'], 'bloodshed': ['0.864', 'anger'], 'angriest': ['0.864', 'anger'], 'smite': ['0.862', 'anger'], 'brawl': ['0.861', 'anger'], 'malicious': ['0.859', 'anger'], 'tirade': ['0.859', 'anger'], 'assault': ['0.859', 'anger'], 'hostility': ['0.859', 'anger'], 'explode': ['0.859', 'anger'], 'assassination': ['0.859', 'anger'], 'strangle': ['0.859', 'anger'], 'loathsome': ['0.857', 'anger'], 'murderous': ['0.853', 'anger'], 'attack': ['0.853', 'anger'], 'hell': ['0.853', 'anger'], 'malice': ['0.852', 'anger'], 'terrorism': ['0.851', 'anger'], 'beating': ['0.849', 'anger'], 'desecration': ['0.848', 'anger'], 'pissingmeoff': ['0.848', 'anger'], 'outrage': ['0.848', 'anger'], 'destroying': ['0.844', 'anger'], 'irate': ['0.844', 'anger'], 'infuriate': ['0.844', 'anger'], 'stab': ['0.844', 'anger'], 'violent': ['0.844', 'anger'], 'tumultuous': ['0.844', 'anger'], 'abomination': ['0.844', 'anger'], 'slaughter': ['0.844', 'anger'], 'obliterate': ['0.843', 'anger'], 'belligerent': ['0.841', 'anger'], 'dumbbitch': ['0.841', 'anger'], 'detest': ['0.838', 'anger'], 'hostilities': ['0.837', 'anger'], 'prick': ['0.835', 'anger'], 'torture': ['0.833', 'anger'], 'rabid': ['0.833', 'anger'], 'rampage': ['0.833', 'anger'], 'horrid': ['0.833', 'anger'], 'cruelty': ['0.833', 'anger'], 'despicable': ['0.828', 'anger'], 'tyrannical': ['0.828', 'anger'], 'demonic': ['0.828', 'anger'], 'hating': ['0.828', 'anger'], 'ragemode': ['0.828', 'anger'], 'hate': ['0.828', 'anger'], 'satanic': ['0.828', 'anger'], 'ruinous': ['0.825', 'anger'], 'condemn': ['0.825', 'anger'], 'dickhead': ['0.824', 'anger'], 'demolish': ['0.824', 'anger'], 'angry': ['0.824', 'anger'], 'riots': ['0.824', 'anger'], 'extermination': ['0.824', 'anger'], 'livid': ['0.821', 'anger'], 'madman': ['0.820', 'anger'], 'vindictive': ['0.819', 'anger'], 'terrorist': ['0.818', 'anger'], 'threaten': ['0.818', 'anger'], 'hateyou': ['0.818', 'anger'], 'effyou': ['0.818', 'anger'], 'ferocity': ['0.818', 'anger'], 'venomous': ['0.818', 'anger'], 'abhor': ['0.816', 'anger'], 'savage': ['0.814', 'anger'], 'atrocity': ['0.814', 'anger'], 'carnage': ['0.814', 'anger'], 'angrytweet': ['0.812', 'anger'], 'barbaric': ['0.812', 'anger'], 'vendetta': ['0.812', 'anger'], 'destroyer': ['0.812', 'anger'], 'pissedoff': ['0.812', 'anger'], 'abuse': ['0.812', 'anger'], 'fuming': ['0.812', 'anger'], 'pissesmeoff': ['0.812', 'anger'], 'berserk': ['0.812', 'anger'], 'fierce': ['0.812', 'anger'], 'fucksake': ['0.812', 'anger'], 'tyrant': ['0.812', 'anger'], 'anger': ['0.811', 'anger'], 'pieceofshit': ['0.810', 'anger'], 'homicide': ['0.803', 'anger'], 'slam': ['0.803', 'anger'], 'punching': ['0.803', 'anger'], 'bitch': ['0.803', 'anger'], 'fights': ['0.803', 'anger'], 'punched': ['0.803', 'anger'], 'ruthless': ['0.797', 'anger'], 'destructive': ['0.797', 'anger'], 'villainous': ['0.797', 'anger'], 'slap': ['0.791', 'anger'], 'yelling': ['0.788', 'anger'], 'ragetweet': ['0.788', 'anger'], 'punishing': ['0.788', 'anger'], 'diabolical': ['0.788', 'anger'], 'riot': ['0.788', 'anger'], 'growthefuckup': ['0.788', 'anger'], 'destroyed': ['0.788', 'anger'], 'retaliatory': ['0.788', 'anger'], 'slaughterhouse': ['0.788', 'anger'], 'manslaughter': ['0.783', 'anger'], 'clash': ['0.783', 'anger'], 'detonation': ['0.781', 'anger'], 'sinister': ['0.781', 'anger'], 'hellish': ['0.781', 'anger'], 'quarrel': ['0.781', 'anger'], 'bloody': ['0.781', 'anger'], 'loath': ['0.781', 'anger'], 'treacherous': ['0.779', 'anger'], 'fumin': ['0.779', 'anger'], 'hateeee': ['0.779', 'anger'], 'accusing': ['0.779', 'anger'], 'horrific': ['0.773', 'anger'], 'revulsion': ['0.773', 'anger'], 'madder': ['0.773', 'anger'], 'retaliate': ['0.773', 'anger'], 'scorn': ['0.769', 'anger'], 'deplorable': ['0.766', 'anger'], 'bomb': ['0.766', 'anger'], 'resent': ['0.765', 'anger'], 'devastation': ['0.765', 'anger'], 'anarchist': ['0.765', 'anger'], 'firestorm': ['0.765', 'anger'], 'contemptible': ['0.764', 'anger'], 'shittest': ['0.760', 'anger'], 'smash': ['0.758', 'anger'], 'cruel': ['0.758', 'anger'], 'soangry': ['0.758', 'anger'], 'rant': ['0.758', 'anger'], 'deadly': ['0.758', 'anger'], 'outburst': ['0.757', 'anger'], 'snarl': ['0.754', 'anger'], 'offend': ['0.750', 'anger'], 'crazed': ['0.750', 'anger'], 'revolting': ['0.750', 'anger'], 'aggravating': ['0.750', 'anger'], 'horror': ['0.750', 'anger'], 'despise': ['0.750', 'anger'], 'dontmesswithme': ['0.750', 'anger'], 'stfu': ['0.750', 'anger'], 'growling': ['0.750', 'anger'], 'profane': ['0.750', 'anger'], 'vulgarity': ['0.750', 'anger'], 'douchebags': ['0.750', 'anger'], 'fuckedoff': ['0.742', 'anger'], 'violence': ['0.742', 'anger'], 'molestation': ['0.742', 'anger'], 'screaming': ['0.742', 'anger'], 'erupt': ['0.742', 'anger'], 'horrible': ['0.742', 'anger'], 'threat': ['0.742', 'anger'], 'bastards': ['0.741', 'anger'], 'revenge': ['0.738', 'anger'], 'catastrophe': ['0.735', 'anger'], 'menacing': ['0.735', 'anger'], 'damn': ['0.735', 'anger'], 'demon': ['0.735', 'anger'], 'crushing': ['0.735', 'anger'], 'thrash': ['0.734', 'anger'], 'riotous': ['0.734', 'anger'], 'fedup': ['0.734', 'anger'], 'deplore': ['0.734', 'anger'], 'warfare': ['0.734', 'anger'], 'argue': ['0.734', 'anger'], 'vehement': ['0.734', 'anger'], 'persecute': ['0.734', 'anger'], 'flog': ['0.734', 'anger'], 'revolt': ['0.734', 'anger'], 'altercation': ['0.729', 'anger'], 'warlike': ['0.728', 'anger'], 'shitday': ['0.727', 'anger'], 'castrate': ['0.727', 'anger'], 'mutiny': ['0.727', 'anger'], 'sabotage': ['0.727', 'anger'], 'malevolent': ['0.721', 'anger'], 'strike': ['0.721', 'anger'], 'disaster': ['0.721', 'anger'], 'disastrous': ['0.720', 'anger'], 'madden': ['0.719', 'anger'], 'scream': ['0.719', 'anger'], 'arseholes': ['0.719', 'anger'], 'brutal': ['0.719', 'anger'], 'horseshit': ['0.719', 'anger'], 'bastarding': ['0.719', 'anger'], 'tumult': ['0.719', 'anger'], 'disdain': ['0.719', 'anger'], 'devil': ['0.719', 'anger'], 'slay': ['0.719', 'anger'], 'aggravates': ['0.719', 'anger'], 'treachery': ['0.719', 'anger'], 'vermin': ['0.719', 'anger'], 'scorching': ['0.719', 'anger'], 'choke': ['0.719', 'anger'], 'spiteful': ['0.719', 'anger'], 'mutilation': ['0.714', 'anger'], 'mangle': ['0.714', 'anger'], 'criminal': ['0.714', 'anger'], 'anarchism': ['0.714', 'anger'], 'punch': ['0.713', 'anger'], 'denunciation': ['0.713', 'anger'], 'holocaust': ['0.712', 'anger'], 'virulence': ['0.712', 'anger'], 'fatal': ['0.712', 'anger'], 'blasphemous': ['0.712', 'anger'], 'hurting': ['0.712', 'anger'], 'dontlikeyou': ['0.712', 'anger'], 'dumbasses': ['0.712', 'anger'], 'battled': ['0.712', 'anger'], 'crucifixion': ['0.712', 'anger'], 'irritated': ['0.706', 'anger'], 'evil': ['0.706', 'anger'], 'atrocious': ['0.706', 'anger'], 'deranged': ['0.706', 'anger'], 'kidnap': ['0.703', 'anger'], 'aggravated': ['0.703', 'anger'], 'assassin': ['0.703', 'anger'], 'scolding': ['0.703', 'anger'], 'dicks': ['0.703', 'anger'], 'slayer': ['0.703', 'anger'], 'intimidation': ['0.703', 'anger'], 'persecution': ['0.703', 'anger'], 'aggression': ['0.702', 'anger'], 'armed': ['0.700', 'anger'], 'poison': ['0.697', 'anger'], 'venom': ['0.697', 'anger'], 'snarling': ['0.697', 'anger'], 'battle': ['0.697', 'anger'], 'disgruntled': ['0.693', 'anger'], 'assailant': ['0.691', 'anger'], 'resentment': ['0.691', 'anger'], 'insidious': ['0.691', 'anger'], 'lynch': ['0.690', 'anger'], 'contemptuous': ['0.690', 'anger'], 'infanticide': ['0.688', 'anger'], 'imprisonment': ['0.688', 'anger'], 'temper': ['0.688', 'anger'], 'terrible': ['0.688', 'anger'], 'mad': ['0.688', 'anger'], 'lunatic': ['0.688', 'anger'], 'domination': ['0.688', 'anger'], 'peeved': ['0.688', 'anger'], 'makesmemad': ['0.688', 'anger'], 'bully': ['0.688', 'anger'], 'curse': ['0.688', 'anger'], 'disparage': ['0.688', 'anger'], 'volatility': ['0.687', 'anger'], 'eradication': ['0.685', 'anger'], 'devastate': ['0.682', 'anger'], 'tantrum': ['0.682', 'anger'], 'scoundrel': ['0.682', 'anger'], 'eradicate': ['0.682', 'anger'], 'aggressively': ['0.680', 'anger'], 'agitation': ['0.680', 'anger'], 'dictatorship': ['0.676', 'anger'], 'irritates': ['0.676', 'anger'], 'profanity': ['0.673', 'anger'], 'shot': ['0.672', 'anger'], 'expletive': ['0.672', 'anger'], 'nasty': ['0.672', 'anger'], 'crime': ['0.672', 'anger'], 'poisonous': ['0.672', 'anger'], 'corrupting': ['0.672', 'anger'], 'dastardly': ['0.672', 'anger'], 'shoot': ['0.672', 'anger'], 'shove': ['0.672', 'anger'], 'condemnation': ['0.672', 'anger'], 'aggravation': ['0.672', 'anger'], 'wreak': ['0.672', 'anger'], 'egregious': ['0.672', 'anger'], 'contempt': ['0.672', 'anger'], 'crushed': ['0.672', 'anger'], 'harmful': ['0.672', 'anger'], 'cruelly': ['0.672', 'anger'], 'maniac': ['0.670', 'anger'], 'combat': ['0.667', 'anger'], 'aggressive': ['0.667', 'anger'], 'hit': ['0.667', 'anger'], 'fight': ['0.667', 'anger'], 'shout': ['0.667', 'anger'], 'cutthroat': ['0.667', 'anger'], 'irritable': ['0.667', 'anger'], 'odious': ['0.667', 'anger'], 'shooting': ['0.667', 'anger'], 'hateeveryone': ['0.667', 'anger'], 'kick': ['0.667', 'anger'], 'eruption': ['0.667', 'anger'], 'enemy': ['0.667', 'anger'], 'punished': ['0.662', 'anger'], 'ambush': ['0.661', 'anger'], 'yell': ['0.661', 'anger'], 'harass': ['0.659', 'anger'], 'incense': ['0.656', 'anger'], 'gore': ['0.656', 'anger'], 'malignant': ['0.656', 'anger'], 'grudge': ['0.656', 'anger'], 'antichrist': ['0.656', 'anger'], 'aggressor': ['0.656', 'anger'], 'expel': ['0.656', 'anger'], 'destruction': ['0.656', 'anger'], 'cranky': ['0.653', 'anger'], 'growl': ['0.652', 'anger'], 'slave': ['0.652', 'anger'], 'spank': ['0.652', 'anger'], 'denounce': ['0.652', 'anger'], 'reprisal': ['0.652', 'anger'], 'insulting': ['0.652', 'anger'], 'clashing': ['0.652', 'anger'], 'insurrection': ['0.652', 'anger'], 'offended': ['0.652', 'anger'], 'animosity': ['0.652', 'anger'], 'growls': ['0.649', 'anger'], 'executioner': ['0.644', 'anger'], 'twat': ['0.644', 'anger'], 'doomsday': ['0.643', 'anger'], 'arson': ['0.641', 'anger'], 'grr': ['0.641', 'anger'], 'daemon': ['0.641', 'anger'], 'spat': ['0.641', 'anger'], 'obscenity': ['0.641', 'anger'], 'havoc': ['0.641', 'anger'], 'shackle': ['0.641', 'anger'], 'accused': ['0.641', 'anger'], 'feud': ['0.641', 'anger'], 'expulsion': ['0.641', 'anger'], 'indignant': ['0.641', 'anger'], 'reprimand': ['0.641', 'anger'], 'inexcusable': ['0.641', 'anger'], 'bombard': ['0.641', 'anger'], 'somad': ['0.637', 'anger'], 'spanking': ['0.636', 'anger'], 'suicidal': ['0.636', 'anger'], 'anarchy': ['0.636', 'anger'], 'combatant': ['0.636', 'anger'], 'hanging': ['0.636', 'anger'], 'poisoned': ['0.636', 'anger'], 'frustrated': ['0.636', 'anger'], 'wound': ['0.636', 'anger'], 'glaring': ['0.636', 'anger'], 'batter': ['0.636', 'anger'], 'disgusting': ['0.636', 'anger'], 'kicking': ['0.636', 'anger'], 'inflict': ['0.633', 'anger'], 'wrecked': ['0.633', 'anger'], 'grievous': ['0.632', 'anger'], 'prosecute': ['0.630', 'anger'], 'agitated': ['0.630', 'anger'], 'cheat': ['0.630', 'anger'], 'swastika': ['0.627', 'anger'], 'raid': ['0.625', 'anger'], 'cursing': ['0.625', 'anger'], 'harassing': ['0.625', 'anger'], 'provocation': ['0.625', 'anger'], 'strife': ['0.625', 'anger'], 'suffocation': ['0.625', 'anger'], 'defamatory': ['0.625', 'anger'], 'scourge': ['0.625', 'anger'], 'injure': ['0.625', 'anger'], 'enslaved': ['0.625', 'anger'], 'indict': ['0.625', 'anger'], 'betray': ['0.625', 'anger'], 'thundering': ['0.625', 'anger'], 'arsehole': ['0.624', 'anger'], 'jerk': ['0.621', 'anger'], 'insane': ['0.621', 'anger'], 'retaliation': ['0.621', 'anger'], 'deprivation': ['0.621', 'anger'], 'convict': ['0.621', 'anger'], 'theft': ['0.621', 'anger'], 'irritate': ['0.621', 'anger'], 'fiend': ['0.621', 'anger'], 'cussed': ['0.619', 'anger'], 'turmoil': ['0.618', 'anger'], 'smack': ['0.615', 'anger'], 'retribution': ['0.614', 'anger'], 'slavery': ['0.609', 'anger'], 'irritability': ['0.609', 'anger'], 'bitterly': ['0.609', 'anger'], 'battery': ['0.609', 'anger'], 'antagonism': ['0.609', 'anger'], 'twats': ['0.609', 'anger'], 'oppressor': ['0.609', 'anger'], 'injurious': ['0.609', 'anger'], 'intolerable': ['0.609', 'anger'], 'gang': ['0.609', 'anger'], 'rebellion': ['0.609', 'anger'], 'collision': ['0.609', 'anger'], 'adverse': ['0.609', 'anger'], 'disgraced': ['0.608', 'anger'], 'revolution': ['0.606', 'anger'], 'diatribe': ['0.606', 'anger'], 'asshole': ['0.606', 'anger'], 'ranting': ['0.606', 'anger'], 'thug': ['0.606', 'anger'], 'antagonistic': ['0.606', 'anger'], 'blast': ['0.606', 'anger'], 'sickening': ['0.606', 'anger'], 'irritating': ['0.606', 'anger'], 'irks': ['0.606', 'anger'], 'bombardment': ['0.606', 'anger'], 'discrimination': ['0.606', 'anger'], 'frustrate': ['0.604', 'anger'], 'oppression': ['0.603', 'anger'], 'insult': ['0.603', 'anger'], 'tiredofit': ['0.603', 'anger'], 'manipulation': ['0.603', 'anger'], 'bigot': ['0.603', 'anger'], 'tension': ['0.603', 'anger'], 'hurtful': ['0.603', 'anger'], 'disgust': ['0.602', 'anger'], 'spite': ['0.600', 'anger'], 'intrusive': ['0.598', 'anger'], 'harshness': ['0.597', 'anger'], 'slur': ['0.596', 'anger'], 'wretch': ['0.594', 'anger'], 'invasion': ['0.594', 'anger'], 'morbidity': ['0.594', 'anger'], 'assail': ['0.594', 'anger'], 'tempest': ['0.594', 'anger'], 'miserable': ['0.594', 'anger'], 'puncture': ['0.594', 'anger'], 'casualty': ['0.594', 'anger'], 'bitterness': ['0.594', 'anger'], 'inferno': ['0.594', 'anger'], 'storming': ['0.594', 'anger'], 'consternation': ['0.592', 'anger'], 'raving': ['0.591', 'anger'], 'guilty': ['0.591', 'anger'], 'depraved': ['0.591', 'anger'], 'immoral': ['0.591', 'anger'], 'forcibly': ['0.591', 'anger'], 'overpowering': ['0.591', 'anger'], 'guillotine': ['0.591', 'anger'], 'recalcitrant': ['0.588', 'anger'], 'accursed': ['0.588', 'anger'], 'invader': ['0.588', 'anger'], 'scare': ['0.588', 'anger'], 'screwed': ['0.588', 'anger'], 'soannoyed': ['0.588', 'anger'], 'jealousy': ['0.587', 'anger'], 'indignation': ['0.587', 'anger'], 'vexed': ['0.586', 'anger'], 'confront': ['0.582', 'anger'], 'brute': ['0.581', 'anger'], 'throttle': ['0.579', 'anger'], 'bickering': ['0.578', 'anger'], 'coup': ['0.578', 'anger'], 'defiant': ['0.578', 'anger'], 'criminality': ['0.578', 'anger'], 'provoking': ['0.578', 'anger'], 'conflict': ['0.578', 'anger'], 'revolver': ['0.578', 'anger'], 'butcher': ['0.578', 'anger'], 'lash': ['0.578', 'anger'], 'incarceration': ['0.578', 'anger'], 'contentious': ['0.578', 'anger'], 'shutit': ['0.578', 'anger'], 'yousuck': ['0.578', 'anger'], 'damage': ['0.578', 'anger'], 'wreck': ['0.578', 'anger'], 'pillage': ['0.578', 'anger'], 'shutup': ['0.578', 'anger'], 'blaze': ['0.578', 'anger'], 'slut': ['0.578', 'anger'], 'cancer': ['0.577', 'anger'], 'blasphemy': ['0.576', 'anger'], 'disturbance': ['0.576', 'anger'], 'dontmess': ['0.576', 'anger'], 'standoff': ['0.576', 'anger'], 'pernicious': ['0.576', 'anger'], 'alienation': ['0.576', 'anger'], 'gun': ['0.576', 'anger'], 'discord': ['0.576', 'anger'], 'grope': ['0.576', 'anger'], 'chaotic': ['0.576', 'anger'], 'frustration': ['0.576', 'anger'], 'gory': ['0.576', 'anger'], 'condescension': ['0.576', 'anger'], 'discriminate': ['0.576', 'anger'], 'friggen': ['0.575', 'anger'], 'death': ['0.574', 'anger'], 'lunacy': ['0.574', 'anger'], 'jab': ['0.574', 'anger'], 'oppressive': ['0.574', 'anger'], 'cursed': ['0.574', 'anger'], 'monstrosity': ['0.574', 'anger'], 'scandalous': ['0.574', 'anger'], 'sneer': ['0.574', 'anger'], 'shit': ['0.573', 'anger'], 'slash': ['0.571', 'anger'], 'disparaging': ['0.571', 'anger'], 'unfair': ['0.571', 'anger'], 'gallows': ['0.570', 'anger'], 'escalate': ['0.569', 'anger'], 'intolerant': ['0.564', 'anger'], 'lawlessness': ['0.563', 'anger'], 'grouchy': ['0.562', 'anger'], 'bellows': ['0.562', 'anger'], 'traitor': ['0.562', 'anger'], 'frightful': ['0.562', 'anger'], 'perdition': ['0.562', 'anger'], 'slander': ['0.562', 'anger'], 'taunt': ['0.562', 'anger'], 'invade': ['0.562', 'anger'], 'wrangling': ['0.562', 'anger'], 'malign': ['0.562', 'anger'], 'bluddy': ['0.562', 'anger'], 'arghh': ['0.562', 'anger'], 'dreadful': ['0.562', 'anger'], 'bearish': ['0.562', 'anger'], 'derogatory': ['0.562', 'anger'], 'glare': ['0.562', 'anger'], 'deceived': ['0.562', 'anger'], 'torpedo': ['0.562', 'anger'], 'retards': ['0.562', 'anger'], 'beast': ['0.562', 'anger'], 'cross': ['0.561', 'anger'], 'hurt': ['0.561', 'anger'], 'banshee': ['0.561', 'anger'], 'uncontrollable': ['0.561', 'anger'], 'shatter': ['0.561', 'anger'], 'jeopardize': ['0.561', 'anger'], 'devastating': ['0.561', 'anger'], 'conflagration': ['0.561', 'anger'], 'thief': ['0.561', 'anger'], 'idiots': ['0.561', 'anger'], 'fits': ['0.561', 'anger'], 'grating': ['0.561', 'anger'], 'rave': ['0.561', 'anger'], 'dissension': ['0.561', 'anger'], 'betrayal': ['0.561', 'anger'], 'disturbed': ['0.559', 'anger'], 'subjugation': ['0.559', 'anger'], 'stomped': ['0.557', 'anger'], 'grab': ['0.557', 'anger'], 'ticked': ['0.556', 'anger'], 'grievance': ['0.556', 'anger'], 'masochism': ['0.556', 'anger'], 'defiance': ['0.552', 'anger'], 'blackmail': ['0.550', 'anger'], 'offensive': ['0.549', 'anger'], 'decry': ['0.548', 'anger'], 'sin': ['0.547', 'anger'], 'violation': ['0.547', 'anger'], 'confine': ['0.547', 'anger'], 'fustrated': ['0.547', 'anger'], 'overbearing': ['0.547', 'anger'], 'deceive': ['0.547', 'anger'], 'misery': ['0.547', 'anger'], 'rebel': ['0.547', 'anger'], 'punishment': ['0.547', 'anger'], 'firearms': ['0.547', 'anger'], 'darkside': ['0.547', 'anger'], 'arghhhh': ['0.547', 'anger'], 'disparity': ['0.547', 'anger'], 'frustrates': ['0.547', 'anger'], 'disgraceful': ['0.547', 'anger'], 'shrill': ['0.547', 'anger'], 'ire': ['0.547', 'anger'], 'preposterous': ['0.547', 'anger'], 'hadenough': ['0.546', 'anger'], 'stolen': ['0.546', 'anger'], 'prejudice': ['0.545', 'anger'], 'annoyin': ['0.545', 'anger'], 'humiliate': ['0.545', 'anger'], 'resentful': ['0.545', 'anger'], 'conspirator': ['0.545', 'anger'], 'callous': ['0.545', 'anger'], 'ruined': ['0.545', 'anger'], 'adversary': ['0.545', 'anger'], 'menace': ['0.545', 'anger'], 'wanker': ['0.545', 'anger'], 'antagonist': ['0.545', 'anger'], 'skirmish': ['0.545', 'anger'], 'tackle': ['0.545', 'anger'], 'heated': ['0.545', 'anger'], 'foul': ['0.545', 'anger'], 'argument': ['0.545', 'anger'], 'sting': ['0.544', 'anger'], 'grumble': ['0.544', 'anger'], 'robbery': ['0.544', 'anger'], 'entangled': ['0.544', 'anger'], 'outcry': ['0.544', 'anger'], 'irreconcilable': ['0.543', 'anger'], 'resistance': ['0.543', 'anger'], 'obstructive': ['0.542', 'anger'], 'dismay': ['0.540', 'anger'], 'mob': ['0.538', 'anger'], 'juststop': ['0.537', 'anger'], 'badness': ['0.536', 'anger'], 'ridicule': ['0.534', 'anger'], 'incendiary': ['0.533', 'anger'], 'flares': ['0.532', 'anger'], 'uprising': ['0.531', 'anger'], 'twofaced': ['0.531', 'anger'], 'exacerbation': ['0.531', 'anger'], 'dispute': ['0.531', 'anger'], 'whip': ['0.531', 'anger'], 'communism': ['0.531', 'anger'], 'prejudicial': ['0.531', 'anger'], 'intruder': ['0.531', 'anger'], 'belittle': ['0.531', 'anger'], 'confinement': ['0.531', 'anger'], 'unbridled': ['0.531', 'anger'], 'allegation': ['0.531', 'anger'], 'reckless': ['0.531', 'anger'], 'degeneracy': ['0.531', 'anger'], 'dictatorial': ['0.530', 'anger'], 'unjustifiable': ['0.530', 'anger'], 'bigoted': ['0.530', 'anger'], 'unjust': ['0.530', 'anger'], 'hassle': ['0.530', 'anger'], 'perversion': ['0.530', 'anger'], 'offender': ['0.530', 'anger'], 'fiesty': ['0.530', 'anger'], 'tackled': ['0.530', 'anger'], 'dissonance': ['0.530', 'anger'], 'renegade': ['0.529', 'anger'], 'hot': ['0.529', 'anger'], 'prison': ['0.529', 'anger'], 'cantstandit': ['0.529', 'anger'], 'trespass': ['0.529', 'anger'], 'suicide': ['0.521', 'anger'], 'annoy': ['0.520', 'anger'], 'leavemealone': ['0.517', 'anger'], 'dishonest': ['0.516', 'anger'], 'depravity': ['0.516', 'anger'], 'distrust': ['0.516', 'anger'], 'broil': ['0.516', 'anger'], 'idiotic': ['0.516', 'anger'], 'treason': ['0.516', 'anger'], 'venting': ['0.516', 'anger'], 'tortious': ['0.516', 'anger'], 'duress': ['0.516', 'anger'], 'criticize': ['0.516', 'anger'], 'grrr': ['0.516', 'anger'], 'inimical': ['0.516', 'anger'], 'disrespectful': ['0.516', 'anger'], 'cretins': ['0.516', 'anger'], 'prisoner': ['0.516', 'anger'], 'divorce': ['0.516', 'anger'], 'chaos': ['0.515', 'anger'], 'coercion': ['0.515', 'anger'], 'unforgiving': ['0.515', 'anger'], 'unkind': ['0.515', 'anger'], 'frustrating': ['0.515', 'anger'], 'bile': ['0.515', 'anger'], 'unleash': ['0.515', 'anger'], 'argumentation': ['0.515', 'anger'], 'jerks': ['0.515', 'anger'], 'pow': ['0.515', 'anger'], 'grump': ['0.515', 'anger'], 'hangry': ['0.515', 'anger'], 'victimized': ['0.515', 'anger'], 'poachers': ['0.515', 'anger'], 'scold': ['0.515', 'anger'], 'poaching': ['0.515', 'anger'], 'roar': ['0.515', 'anger'], 'tussle': ['0.514', 'anger'], 'bane': ['0.511', 'anger'], 'repudiation': ['0.510', 'anger'], 'accusation': ['0.510', 'anger'], 'enmity': ['0.510', 'anger'], 'banish': ['0.509', 'anger'], 'disfigured': ['0.508', 'anger'], 'storm': ['0.507', 'anger'], 'fear': ['0.500', 'anger'], 'crazy': ['0.500', 'anger'], 'anguish': ['0.500', 'anger'], 'confined': ['0.500', 'anger'], 'scoff': ['0.500', 'anger'], 'shun': ['0.500', 'anger'], 'derogation': ['0.500', 'anger'], 'banished': ['0.500', 'anger'], 'hammering': ['0.500', 'anger'], 'brunt': ['0.500', 'anger'], 'possessed': ['0.500', 'anger'], 'nobodycares': ['0.500', 'anger'], 'frenzied': ['0.500', 'anger'], 'ordeal': ['0.500', 'anger'], 'delusional': ['0.500', 'anger'], 'reject': ['0.500', 'anger'], 'obstruct': ['0.500', 'anger'], 'foaming': ['0.500', 'anger'], 'intractable': ['0.500', 'anger'], 'bout': ['0.500', 'anger'], 'brazen': ['0.500', 'anger'], 'patronising': ['0.500', 'anger'], 'rejects': ['0.500', 'anger'], 'fervor': ['0.500', 'anger'], 'dominate': ['0.500', 'anger'], 'derision': ['0.500', 'anger'], 'spear': ['0.500', 'anger'], 'suppression': ['0.500', 'anger'], 'animus': ['0.500', 'anger'], 'unruly': ['0.500', 'anger'], 'disrupting': ['0.500', 'anger'], 'malpractice': ['0.500', 'anger'], 'defy': ['0.500', 'anger'], 'injustice': ['0.500', 'anger'], 'antithesis': ['0.500', 'anger'], 'getoveryourself': ['0.495', 'anger'], 'toughness': ['0.492', 'anger'], 'stupidpeople': ['0.492', 'anger'], 'madness': ['0.491', 'anger'], 'sinful': ['0.491', 'anger'], 'oppress': ['0.490', 'anger'], 'avarice': ['0.490', 'anger'], 'revoke': ['0.485', 'anger'], 'incest': ['0.485', 'anger'], 'smuggler': ['0.485', 'anger'], 'avenger': ['0.485', 'anger'], 'disapproved': ['0.485', 'anger'], 'demand': ['0.485', 'anger'], 'stayaway': ['0.485', 'anger'], 'claw': ['0.485', 'anger'], 'infraction': ['0.485', 'anger'], 'cutting': ['0.485', 'anger'], 'pervert': ['0.485', 'anger'], 'fricking': ['0.485', 'anger'], 'anathema': ['0.485', 'anger'], 'annoyed': ['0.485', 'anger'], 'disobedient': ['0.485', 'anger'], 'alienate': ['0.485', 'anger'], 'disservice': ['0.485', 'anger'], 'abolish': ['0.485', 'anger'], 'inhuman': ['0.485', 'anger'], 'dissident': ['0.485', 'anger'], 'complaint': ['0.485', 'anger'], 'usurp': ['0.485', 'anger'], 'obnoxious': ['0.484', 'anger'], 'deceit': ['0.484', 'anger'], 'disgrace': ['0.484', 'anger'], 'opposed': ['0.484', 'anger'], 'renounce': ['0.484', 'anger'], 'litigious': ['0.484', 'anger'], 'imprisoned': ['0.484', 'anger'], 'mocking': ['0.484', 'anger'], 'blame': ['0.484', 'anger'], 'penalty': ['0.484', 'anger'], 'thresh': ['0.484', 'anger'], 'upheaval': ['0.484', 'anger'], 'restrain': ['0.484', 'anger'], 'strained': ['0.484', 'anger'], 'sucker': ['0.484', 'anger'], 'rivalry': ['0.484', 'anger'], 'oust': ['0.484', 'anger'], 'suspicious': ['0.484', 'anger'], 'turbulence': ['0.483', 'anger'], 'pound': ['0.481', 'anger'], 'coldness': ['0.477', 'anger'], 'ungrateful': ['0.472', 'anger'], 'battalion': ['0.471', 'anger'], 'stoopid': ['0.471', 'anger'], 'adversity': ['0.470', 'anger'], 'ram': ['0.470', 'anger'], 'gall': ['0.470', 'anger'], 'infidel': ['0.470', 'anger'], 'annoying': ['0.470', 'anger'], 'friction': ['0.470', 'anger'], 'hostage': ['0.470', 'anger'], 'jealous': ['0.470', 'anger'], 'accuser': ['0.470', 'anger'], 'feudalism': ['0.470', 'anger'], 'subversion': ['0.470', 'anger'], 'armament': ['0.470', 'anger'], 'dispossessed': ['0.470', 'anger'], 'distress': ['0.469', 'anger'], 'intolerance': ['0.469', 'anger'], 'subversive': ['0.469', 'anger'], 'opposition': ['0.469', 'anger'], 'exile': ['0.469', 'anger'], 'plunder': ['0.469', 'anger'], 'recidivism': ['0.469', 'anger'], 'objection': ['0.469', 'anger'], 'steal': ['0.469', 'anger'], 'offense': ['0.469', 'anger'], 'complain': ['0.469', 'anger'], 'huff': ['0.469', 'anger'], 'simmer': ['0.469', 'anger'], 'selfish': ['0.469', 'anger'], 'backoff': ['0.469', 'anger'], 'incite': ['0.469', 'anger'], 'angermanagement': ['0.469', 'anger'], 'perpetrator': ['0.469', 'anger'], 'disobey': ['0.469', 'anger'], 'disruption': ['0.469', 'anger'], 'smother': ['0.469', 'anger'], 'injury': ['0.469', 'anger'], 'selfishness': ['0.469', 'anger'], 'insanity': ['0.469', 'anger'], 'stifled': ['0.469', 'anger'], 'thump': ['0.469', 'anger'], 'dislike': ['0.469', 'anger'], 'areyoukidding': ['0.468', 'anger'], 'mug': ['0.467', 'anger'], 'agony': ['0.465', 'anger'], 'arrogant': ['0.461', 'anger'], 'elimination': ['0.456', 'anger'], 'picketing': ['0.456', 'anger'], 'haughty': ['0.456', 'anger'], 'sux': ['0.456', 'anger'], 'vent': ['0.456', 'anger'], 'grated': ['0.455', 'anger'], 'clamor': ['0.455', 'anger'], 'constraint': ['0.455', 'anger'], 'stubbed': ['0.455', 'anger'], 'impermeable': ['0.455', 'anger'], 'illegality': ['0.455', 'anger'], 'dying': ['0.455', 'anger'], 'flagrant': ['0.455', 'anger'], 'idiocy': ['0.455', 'anger'], 'busted': ['0.455', 'anger'], 'crabby': ['0.455', 'anger'], 'illicit': ['0.455', 'anger'], 'veto': ['0.455', 'anger'], 'troublesome': ['0.455', 'anger'], 'mislead': ['0.455', 'anger'], 'bad': ['0.453', 'anger'], 'despotism': ['0.453', 'anger'], 'struggle': ['0.453', 'anger'], 'usurped': ['0.453', 'anger'], 'disagreeing': ['0.453', 'anger'], 'hysterical': ['0.453', 'anger'], 'desist': ['0.453', 'anger'], 'godless': ['0.453', 'anger'], 'suppress': ['0.453', 'anger'], 'disapproving': ['0.453', 'anger'], 'displeased': ['0.453', 'anger'], 'bayonet': ['0.453', 'anger'], 'intense': ['0.453', 'anger'], 'unlawful': ['0.451', 'anger'], 'wrongly': ['0.448', 'anger'], 'repellent': ['0.442', 'anger'], 'psychosis': ['0.441', 'anger'], 'foe': ['0.441', 'anger'], 'wrongful': ['0.441', 'anger'], 'dishonor': ['0.441', 'anger'], 'wasted': ['0.441', 'anger'], 'aversion': ['0.440', 'anger'], 'schism': ['0.439', 'anger'], 'gahhh': ['0.439', 'anger'], 'punitive': ['0.439', 'anger'], 'knuckles': ['0.439', 'anger'], 'upset': ['0.439', 'anger'], 'effigy': ['0.439', 'anger'], 'ultimatum': ['0.439', 'anger'], 'deleterious': ['0.438', 'anger'], 'mucked': ['0.438', 'anger'], 'irritation': ['0.438', 'anger'], 'worthless': ['0.438', 'anger'], 'ransom': ['0.438', 'anger'], 'separatist': ['0.438', 'anger'], 'fugitive': ['0.438', 'anger'], 'deny': ['0.438', 'anger'], 'abandonment': ['0.438', 'anger'], 'stupidity': ['0.438', 'anger'], 'oblivion': ['0.437', 'anger'], 'segregate': ['0.437', 'anger'], 'payback': ['0.436', 'anger'], 'eviction': ['0.435', 'anger'], 'incongruous': ['0.435', 'anger'], 'collusion': ['0.432', 'anger'], 'rob': ['0.431', 'anger'], 'infidelity': ['0.429', 'anger'], 'ravenous': ['0.429', 'anger'], 'overrun': ['0.429', 'anger'], 'incredulous': ['0.426', 'anger'], 'stupidrain': ['0.426', 'anger'], 'martial': ['0.426', 'anger'], 'painful': ['0.426', 'anger'], 'harbinger': ['0.426', 'anger'], 'getoverit': ['0.426', 'anger'], 'rejection': ['0.426', 'anger'], 'defense': ['0.425', 'anger'], 'unsympathetic': ['0.424', 'anger'], 'banger': ['0.424', 'anger'], 'gonorrhea': ['0.424', 'anger'], 'fallacious': ['0.424', 'anger'], 'indecency': ['0.424', 'anger'], 'exasperation': ['0.424', 'anger'], 'fuss': ['0.424', 'anger'], 'concealment': ['0.424', 'anger'], 'powerful': ['0.424', 'anger'], 'fraudulent': ['0.424', 'anger'], 'defraud': ['0.424', 'anger'], 'enforce': ['0.424', 'anger'], 'censor': ['0.424', 'anger'], 'greed': ['0.424', 'anger'], 'disobedience': ['0.424', 'anger'], 'commotion': ['0.424', 'anger'], 'discontent': ['0.424', 'anger'], 'penitentiary': ['0.422', 'anger'], 'nettle': ['0.422', 'anger'], 'duel': ['0.422', 'anger'], 'banishment': ['0.422', 'anger'], 'barb': ['0.422', 'anger'], 'deportation': ['0.422', 'anger'], 'sarcasm': ['0.422', 'anger'], 'penetration': ['0.422', 'anger'], 'bang': ['0.422', 'anger'], 'scar': ['0.422', 'anger'], 'cracked': ['0.422', 'anger'], 'sedition': ['0.422', 'anger'], 'annoyance': ['0.422', 'anger'], 'cur': ['0.422', 'anger'], 'snubbed': ['0.422', 'anger'], 'misrepresented': ['0.422', 'anger'], 'blatant': ['0.420', 'anger'], 'force': ['0.418', 'anger'], 'perverse': ['0.415', 'anger'], 'wring': ['0.415', 'anger'], 'grim': ['0.413', 'anger'], 'bastion': ['0.413', 'anger'], 'sordid': ['0.412', 'anger'], 'nothappy': ['0.412', 'anger'], 'moody': ['0.412', 'anger'], 'tiff': ['0.412', 'anger'], 'surly': ['0.412', 'anger'], 'hunting': ['0.411', 'anger'], 'indenture': ['0.410', 'anger'], 'areyoukiddingme': ['0.409', 'anger'], 'reproach': ['0.409', 'anger'], 'compulsion': ['0.409', 'anger'], 'sham': ['0.409', 'anger'], 'cantwin': ['0.409', 'anger'], 'supremacy': ['0.409', 'anger'], 'disappoint': ['0.409', 'anger'], 'squelch': ['0.409', 'anger'], 'forfeit': ['0.409', 'anger'], 'awful': ['0.409', 'anger'], 'detainee': ['0.409', 'anger'], 'implicate': ['0.409', 'anger'], 'blockade': ['0.409', 'anger'], 'sneak': ['0.409', 'anger'], 'contradict': ['0.409', 'anger'], 'inept': ['0.409', 'anger'], 'lying': ['0.408', 'anger'], 'antipathy': ['0.406', 'anger'], 'delusion': ['0.406', 'anger'], 'unthinkable': ['0.406', 'anger'], 'wop': ['0.406', 'anger'], 'tremor': ['0.406', 'anger'], 'onerous': ['0.406', 'anger'], 'forsaken': ['0.406', 'anger'], 'inequality': ['0.406', 'anger'], 'badger': ['0.406', 'anger'], 'cacophony': ['0.406', 'anger'], 'wrongdoing': ['0.406', 'anger'], 'epidemic': ['0.406', 'anger'], 'rail': ['0.406', 'anger'], 'nuisance': ['0.406', 'anger'], 'scrapie': ['0.406', 'anger'], 'arguments': ['0.404', 'anger'], 'affront': ['0.403', 'anger'], 'traumatic': ['0.402', 'anger'], 'sodding': ['0.400', 'anger'], 'libel': ['0.400', 'anger'], 'annoys': ['0.400', 'anger'], 'soslow': ['0.398', 'anger'], 'watchout': ['0.398', 'anger'], 'frenetic': ['0.397', 'anger'], 'remiss': ['0.397', 'anger'], 'barge': ['0.396', 'anger'], 'fraud': ['0.394', 'anger'], 'howl': ['0.394', 'anger'], 'confiscate': ['0.394', 'anger'], 'boxing': ['0.394', 'anger'], 'nag': ['0.394', 'anger'], 'actionable': ['0.394', 'anger'], 'illegal': ['0.394', 'anger'], 'keyed': ['0.394', 'anger'], 'disrespect': ['0.394', 'anger'], 'dangit': ['0.394', 'anger'], 'extinguish': ['0.394', 'anger'], 'sue': ['0.394', 'anger'], 'untoward': ['0.394', 'anger'], 'rabble': ['0.394', 'anger'], 'unfriendly': ['0.394', 'anger'], 'whatsthepoint': ['0.394', 'anger'], 'brimstone': ['0.392', 'anger'], 'earthquake': ['0.391', 'anger'], 'grrrrr': ['0.391', 'anger'], 'rigged': ['0.391', 'anger'], 'argh': ['0.391', 'anger'], 'pique': ['0.391', 'anger'], 'recklessness': ['0.391', 'anger'], 'dissolution': ['0.391', 'anger'], 'disagree': ['0.389', 'anger'], 'lawsuit': ['0.386', 'anger'], 'despair': ['0.382', 'anger'], 'disused': ['0.382', 'anger'], 'immorality': ['0.382', 'anger'], 'incurable': ['0.379', 'anger'], 'pokes': ['0.379', 'anger'], 'falsification': ['0.379', 'anger'], 'coerce': ['0.379', 'anger'], 'touchy': ['0.379', 'anger'], 'firstworldprobs': ['0.379', 'anger'], 'sore': ['0.379', 'anger'], 'difficulty': ['0.379', 'anger'], 'rifle': ['0.379', 'anger'], 'sizzle': ['0.379', 'anger'], 'picket': ['0.378', 'anger'], 'concussion': ['0.377', 'anger'], 'stuckup': ['0.375', 'anger'], 'pessimism': ['0.375', 'anger'], 'remand': ['0.375', 'anger'], 'pitfall': ['0.375', 'anger'], 'rawr': ['0.375', 'anger'], 'cannon': ['0.375', 'anger'], 'infantile': ['0.375', 'anger'], 'disillusionment': ['0.375', 'anger'], 'sly': ['0.375', 'anger'], 'petpeeve': ['0.375', 'anger'], 'militia': ['0.375', 'anger'], 'faulty': ['0.375', 'anger'], 'inhibit': ['0.371', 'anger'], 'vindicate': ['0.371', 'anger'], 'nepotism': ['0.371', 'anger'], 'distressing': ['0.371', 'anger'], 'schizophrenia': ['0.369', 'anger'], 'skewed': ['0.368', 'anger'], 'disreputable': ['0.368', 'anger'], 'forbidding': ['0.368', 'anger'], 'conquest': ['0.367', 'anger'], 'bark': ['0.365', 'anger'], 'leukemia': ['0.365', 'anger'], 'unhappy': ['0.364', 'anger'], 'burke': ['0.364', 'anger'], 'warrior': ['0.364', 'anger'], 'disapprove': ['0.364', 'anger'], 'challenge': ['0.364', 'anger'], 'retarded': ['0.364', 'anger'], 'belt': ['0.364', 'anger'], 'barks': ['0.364', 'anger'], 'opinionated': ['0.359', 'anger'], 'restriction': ['0.359', 'anger'], 'incompetence': ['0.359', 'anger'], 'polemic': ['0.359', 'anger'], 'loudness': ['0.359', 'anger'], 'paucity': ['0.359', 'anger'], 'controversial': ['0.359', 'anger'], 'aftermath': ['0.359', 'anger'], 'disliked': ['0.359', 'anger'], 'litigate': ['0.359', 'anger'], 'sectarian': ['0.359', 'anger'], 'cad': ['0.359', 'anger'], 'broken': ['0.359', 'anger'], 'interrupting': ['0.358', 'anger'], 'fussy': ['0.357', 'anger'], 'wench': ['0.353', 'anger'], 'remove': ['0.351', 'anger'], 'misbehavior': ['0.351', 'anger'], 'brat': ['0.351', 'anger'], 'gruff': ['0.351', 'anger'], 'scarcity': ['0.350', 'anger'], 'losing': ['0.349', 'anger'], 'timewasters': ['0.348', 'anger'], 'lie': ['0.348', 'anger'], 'stigma': ['0.348', 'anger'], 'untrustworthy': ['0.348', 'anger'], 'deserted': ['0.348', 'anger'], 'disagreement': ['0.348', 'anger'], 'disappointed': ['0.348', 'anger'], 'retract': ['0.348', 'anger'], 'ulcer': ['0.348', 'anger'], 'pest': ['0.348', 'anger'], 'hardened': ['0.348', 'anger'], 'defect': ['0.348', 'anger'], 'bias': ['0.345', 'anger'], 'evade': ['0.344', 'anger'], 'antisocial': ['0.344', 'anger'], 'unreliable': ['0.344', 'anger'], 'misleading': ['0.344', 'anger'], 'stingy': ['0.344', 'anger'], 'anxiety': ['0.344', 'anger'], 'stripped': ['0.344', 'anger'], 'impotence': ['0.344', 'anger'], 'unsettled': ['0.344', 'anger'], 'shaky': ['0.344', 'anger'], 'bothering': ['0.344', 'anger'], 'pirate': ['0.344', 'anger'], 'negation': ['0.344', 'anger'], 'shoddy': ['0.344', 'anger'], 'disclaim': ['0.344', 'anger'], 'deterioration': ['0.344', 'anger'], 'interminable': ['0.343', 'anger'], 'meddle': ['0.341', 'anger'], 'disease': ['0.341', 'anger'], 'warp': ['0.340', 'anger'], 'averse': ['0.338', 'anger'], 'alcoholism': ['0.338', 'anger'], 'infamous': ['0.338', 'anger'], 'row': ['0.337', 'anger'], 'staticky': ['0.336', 'anger'], 'illegitimate': ['0.333', 'anger'], 'encumbrance': ['0.333', 'anger'], 'witchcraft': ['0.333', 'anger'], 'paralyzed': ['0.333', 'anger'], 'ill': ['0.333', 'anger'], 'interrupt': ['0.333', 'anger'], 'scorpion': ['0.333', 'anger'], 'sinner': ['0.331', 'anger'], 'hulk': ['0.329', 'anger'], 'inconsiderate': ['0.329', 'anger'], 'disqualified': ['0.328', 'anger'], 'tighten': ['0.328', 'anger'], 'opponent': ['0.328', 'anger'], 'phony': ['0.328', 'anger'], 'resisting': ['0.328', 'anger'], 'fib': ['0.328', 'anger'], 'spammers': ['0.328', 'anger'], 'dislocated': ['0.328', 'anger'], 'brrr': ['0.328', 'anger'], 'dashed': ['0.328', 'anger'], 'prohibited': ['0.328', 'anger'], 'grumpy': ['0.328', 'anger'], 'victim': ['0.328', 'anger'], 'crusade': ['0.328', 'anger'], 'scapegoat': ['0.328', 'anger'], 'hiss': ['0.328', 'anger'], 'inappropriate': ['0.324', 'anger'], 'haye': ['0.324', 'anger'], 'loss': ['0.324', 'anger'], 'eschew': ['0.324', 'anger'], 'neglected': ['0.324', 'anger'], 'trickery': ['0.324', 'anger'], 'canker': ['0.323', 'anger'], 'crunch': ['0.318', 'anger'], 'criticism': ['0.318', 'anger'], 'queues': ['0.318', 'anger'], 'duplicity': ['0.318', 'anger'], 'muff': ['0.318', 'anger'], 'shriek': ['0.316', 'anger'], 'depreciate': ['0.315', 'anger'], 'dramaqueen': ['0.312', 'anger'], 'carelessness': ['0.312', 'anger'], 'dumps': ['0.312', 'anger'], 'dupe': ['0.312', 'anger'], 'chaff': ['0.312', 'anger'], 'poverty': ['0.312', 'anger'], 'mortality': ['0.312', 'anger'], 'dismissal': ['0.312', 'anger'], 'deflate': ['0.312', 'anger'], 'revving': ['0.309', 'anger'], 'disallowed': ['0.308', 'anger'], 'boisterous': ['0.307', 'anger'], 'thoughtless': ['0.307', 'anger'], 'burial': ['0.304', 'anger'], 'sullen': ['0.303', 'anger'], 'theocratic': ['0.303', 'anger'], 'wince': ['0.303', 'anger'], 'involution': ['0.300', 'anger'], 'stalemate': ['0.297', 'anger'], 'talons': ['0.297', 'anger'], 'hoax': ['0.297', 'anger'], 'depreciated': ['0.297', 'anger'], 'wasteful': ['0.297', 'anger'], 'getyourown': ['0.297', 'anger'], 'senseless': ['0.297', 'anger'], 'depressed': ['0.297', 'anger'], 'taxed': ['0.297', 'anger'], 'misuse': ['0.297', 'anger'], 'paralysis': ['0.295', 'anger'], 'displaced': ['0.294', 'anger'], 'limited': ['0.292', 'anger'], 'disapointment': ['0.290', 'anger'], 'orc': ['0.289', 'anger'], 'ridiculous': ['0.289', 'anger'], 'spine': ['0.288', 'anger'], 'sharpen': ['0.288', 'anger'], 'presumptuous': ['0.288', 'anger'], 'teasing': ['0.288', 'anger'], 'homeless': ['0.288', 'anger'], 'react': ['0.288', 'anger'], 'barrier': ['0.287', 'anger'], 'hoot': ['0.287', 'anger'], 'twitchy': ['0.287', 'anger'], 'myopia': ['0.283', 'anger'], 'incompatible': ['0.281', 'anger'], 'disconnects': ['0.281', 'anger'], 'delinquent': ['0.281', 'anger'], 'contraband': ['0.281', 'anger'], 'lagging': ['0.281', 'anger'], 'shiver': ['0.281', 'anger'], 'agh': ['0.281', 'anger'], 'restitution': ['0.281', 'anger'], 'flexin': ['0.281', 'anger'], 'spam': ['0.281', 'anger'], 'foray': ['0.279', 'anger'], 'noncompliance': ['0.279', 'anger'], 'buffering': ['0.279', 'anger'], 'unfairness': ['0.279', 'anger'], 'troll': ['0.279', 'anger'], 'nether': ['0.278', 'anger'], 'immaturity': ['0.273', 'anger'], 'uncaring': ['0.273', 'anger'], 'bugaboo': ['0.273', 'anger'], 'bogus': ['0.273', 'anger'], 'shock': ['0.269', 'anger'], 'feisty': ['0.269', 'anger'], 'rapping': ['0.266', 'anger'], 'nopoint': ['0.266', 'anger'], 'feminism': ['0.266', 'anger'], 'pry': ['0.266', 'anger'], 'humbug': ['0.266', 'anger'], 'inoperative': ['0.266', 'anger'], 'defendant': ['0.266', 'anger'], 'latent': ['0.266', 'anger'], 'notamorningperson': ['0.266', 'anger'], 'quandary': ['0.266', 'anger'], 'inconvenient': ['0.266', 'anger'], 'bear': ['0.266', 'anger'], 'interrupts': ['0.265', 'anger'], 'fluctuation': ['0.265', 'anger'], 'exaggerate': ['0.263', 'anger'], 'lose': ['0.261', 'anger'], 'stone': ['0.258', 'anger'], 'soldier': ['0.258', 'anger'], 'furnace': ['0.258', 'anger'], 'shoplifting': ['0.258', 'anger'], 'tease': ['0.258', 'anger'], 'patter': ['0.258', 'anger'], 'incompetent': ['0.257', 'anger'], 'indoctrination': ['0.255', 'anger'], 'attentionseeker': ['0.250', 'anger'], 'unfollow': ['0.250', 'anger'], 'nonsense': ['0.250', 'anger'], 'complicate': ['0.250', 'anger'], 'tripping': ['0.250', 'anger'], 'untrue': ['0.250', 'anger'], 'notoriety': ['0.250', 'anger'], 'falsehood': ['0.250', 'anger'], 'mastery': ['0.250', 'anger'], 'socialist': ['0.250', 'anger'], 'skid': ['0.250', 'anger'], 'rocket': ['0.250', 'anger'], 'noisy': ['0.250', 'anger'], 'lawyer': ['0.250', 'anger'], 'pouting': ['0.250', 'anger'], 'cane': ['0.250', 'anger'], 'fenced': ['0.242', 'anger'], 'obstacle': ['0.242', 'anger'], 'dontunderstand': ['0.242', 'anger'], 'detract': ['0.242', 'anger'], 'halter': ['0.242', 'anger'], 'vampire': ['0.242', 'anger'], 'capslock': ['0.242', 'anger'], 'witch': ['0.242', 'anger'], 'ringer': ['0.242', 'anger'], 'frowning': ['0.239', 'anger'], 'saber': ['0.238', 'anger'], 'hunger': ['0.235', 'anger'], 'tariff': ['0.234', 'anger'], 'lava': ['0.234', 'anger'], 'dabbling': ['0.234', 'anger'], 'shell': ['0.234', 'anger'], 'imtryingtosleep': ['0.234', 'anger'], 'rascal': ['0.234', 'anger'], 'recession': ['0.234', 'anger'], 'failing': ['0.234', 'anger'], 'politics': ['0.234', 'anger'], 'wokemeup': ['0.234', 'anger'], 'undesirable': ['0.231', 'anger'], 'versus': ['0.227', 'anger'], 'copycat': ['0.227', 'anger'], 'darkness': ['0.227', 'anger'], 'resign': ['0.227', 'anger'], 'soaked': ['0.226', 'anger'], 'unfulfilled': ['0.225', 'anger'], 'abandoned': ['0.222', 'anger'], 'unattainable': ['0.221', 'anger'], 'owing': ['0.221', 'anger'], 'bankruptcy': ['0.221', 'anger'], 'confusion': ['0.219', 'anger'], 'warden': ['0.219', 'anger'], 'somethingigetalot': ['0.219', 'anger'], 'tool': ['0.219', 'anger'], 'compress': ['0.219', 'anger'], 'misconception': ['0.219', 'anger'], 'whiny': ['0.219', 'anger'], 'unhelpful': ['0.219', 'anger'], 'mosquito': ['0.219', 'anger'], 'twitching': ['0.219', 'anger'], 'nosey': ['0.213', 'anger'], 'adder': ['0.212', 'anger'], 'overpriced': ['0.212', 'anger'], 'shortage': ['0.212', 'anger'], 'melodrama': ['0.212', 'anger'], 'harry': ['0.212', 'anger'], 'possession': ['0.206', 'anger'], 'overplayed': ['0.206', 'anger'], 'desert': ['0.206', 'anger'], 'unlucky': ['0.203', 'anger'], 'unpaid': ['0.203', 'anger'], 'backbone': ['0.203', 'anger'], 'powerless': ['0.203', 'anger'], 'sentence': ['0.203', 'anger'], 'uninvited': ['0.203', 'anger'], 'rook': ['0.203', 'anger'], 'pout': ['0.203', 'anger'], 'arraignment': ['0.203', 'anger'], 'inefficient': ['0.203', 'anger'], 'court': ['0.199', 'anger'], 'endless': ['0.198', 'anger'], 'misstatement': ['0.197', 'anger'], 'delay': ['0.197', 'anger'], 'distracted': ['0.197', 'anger'], 'adverts': ['0.197', 'anger'], 'misunderstanding': ['0.195', 'anger'], 'inadmissible': ['0.191', 'anger'], 'excite': ['0.191', 'anger'], 'lightning': ['0.189', 'anger'], 'mournful': ['0.188', 'anger'], 'preclude': ['0.188', 'anger'], 'incase': ['0.188', 'anger'], 'insecure': ['0.188', 'anger'], 'rating': ['0.182', 'anger'], 'claimant': ['0.182', 'anger'], 'mistress': ['0.182', 'anger'], 'insist': ['0.182', 'anger'], 'pare': ['0.182', 'anger'], 'distracting': ['0.182', 'anger'], 'mutter': ['0.182', 'anger'], 'opium': ['0.180', 'anger'], 'willful': ['0.176', 'anger'], 'deserve': ['0.176', 'anger'], 'insists': ['0.175', 'anger'], 'treat': ['0.175', 'anger'], 'liberate': ['0.172', 'anger'], 'peice': ['0.172', 'anger'], 'excitation': ['0.172', 'anger'], 'misplace': ['0.172', 'anger'], 'hormonal': ['0.172', 'anger'], 'mighty': ['0.172', 'anger'], 'thanksalot': ['0.172', 'anger'], 'indecisive': ['0.172', 'anger'], 'fee': ['0.172', 'anger'], 'gibberish': ['0.172', 'anger'], 'fleece': ['0.172', 'anger'], 'yelp': ['0.168', 'anger'], 'hamstring': ['0.167', 'anger'], 'mule': ['0.167', 'anger'], 'insufficiency': ['0.167', 'anger'], 'insignificant': ['0.167', 'anger'], 'unequal': ['0.167', 'anger'], 'bargaining': ['0.167', 'anger'], 'attentionseekers': ['0.163', 'anger'], 'forearm': ['0.156', 'anger'], 'indifference': ['0.156', 'anger'], 'coop': ['0.156', 'anger'], 'rheumatism': ['0.156', 'anger'], 'attorney': ['0.152', 'anger'], 'uncertain': ['0.152', 'anger'], 'justthebeginning': ['0.152', 'anger'], 'disinformation': ['0.152', 'anger'], 'pretending': ['0.152', 'anger'], 'involvement': ['0.152', 'anger'], 'underpaid': ['0.152', 'anger'], 'bee': ['0.152', 'anger'], 'campaigning': ['0.151', 'anger'], 'hopelessness': ['0.149', 'anger'], 'feeling': ['0.147', 'anger'], 'legalized': ['0.145', 'anger'], 'caution': ['0.145', 'anger'], 'sterling': ['0.141', 'anger'], 'obliging': ['0.141', 'anger'], 'subsidy': ['0.141', 'anger'], 'morals': ['0.141', 'anger'], 'wimpy': ['0.140', 'anger'], 'bummer': ['0.139', 'anger'], 'geez': ['0.136', 'anger'], 'repay': ['0.136', 'anger'], 'blemish': ['0.136', 'anger'], 'misspell': ['0.136', 'anger'], 'surcharge': ['0.136', 'anger'], 'saloon': ['0.136', 'anger'], 'birch': ['0.135', 'anger'], 'noob': ['0.133', 'anger'], 'honk': ['0.133', 'anger'], 'orchestra': ['0.132', 'anger'], 'wireless': ['0.125', 'anger'], 'standstill': ['0.125', 'anger'], 'competitive': ['0.125', 'anger'], 'mosque': ['0.122', 'anger'], 'inattention': ['0.121', 'anger'], 'reversal': ['0.121', 'anger'], 'lace': ['0.118', 'anger'], 'elbow': ['0.117', 'anger'], 'instinctive': ['0.112', 'anger'], 'chant': ['0.111', 'anger'], 'lonely': ['0.109', 'anger'], 'gnome': ['0.109', 'anger'], 'tolerate': ['0.106', 'anger'], 'management': ['0.102', 'anger'], 'advocacy': ['0.100', 'anger'], 'moral': ['0.094', 'anger'], 'roadworks': ['0.091', 'anger'], 'honest': ['0.087', 'anger'], 'gent': ['0.076', 'anger'], 'forgetful': ['0.076', 'anger'], 'liquor': ['0.075', 'anger'], 'money': ['0.074', 'anger'], 'hood': ['0.071', 'anger'], 'curriculum': ['0.063', 'anger'], 'words': ['0.062', 'anger'], 'elf': ['0.061', 'anger'], 'smell': ['0.061', 'anger'], 'opera': ['0.061', 'anger'], 'playful': ['0.061', 'anger'], 'counsellor': ['0.059', 'anger'], 'trumpet': ['0.059', 'anger'], 'nurture': ['0.059', 'anger'], 'asleeep': ['0.059', 'anger'], 'birthplace': ['0.051', 'anger'], 'ribbon': ['0.047', 'anger'], 'youth': ['0.045', 'anger'], 'vote': ['0.045', 'anger'], 'cash': ['0.039', 'anger'], 'wannasleep': ['0.031', 'anger'], 'waffle': ['0.030', 'anger'], 'dame': ['0.030', 'anger'], 'buffet': ['0.029', 'anger'], 'celebrity': ['0.026', 'anger'], 'sisterhood': ['0.015', 'anger'], 'autocorrect': ['0.015', 'anger'], 'musical': ['0.011', 'anger'], 'tree': ['0.000', 'anger']}
di={'torture': ['0.984', 'fear'], 'terrorist': ['0.972', 'fear'], 'horrific': ['0.969', 'fear'], 'terrorism': ['0.969', 'fear'], 'terrorists': ['0.969', 'fear'], 'suicidebombing': ['0.967', 'fear'], 'kill': ['0.962', 'fear'], 'homicidal': ['0.959', 'fear'], 'terror': ['0.953', 'fear'], 'murderer': ['0.953', 'fear'], 'catastrophe': ['0.953', 'fear'], 'annihilate': ['0.953', 'fear'], 'dying': ['0.948', 'fear'], 'war': ['0.942', 'fear'], 'bombing': ['0.938', 'fear'], 'bomb': ['0.935', 'fear'], 'missiles': ['0.934', 'fear'], 'horror': ['0.923', 'fear'], 'horrified': ['0.922', 'fear'], 'terrorize': ['0.922', 'fear'], 'brutality': ['0.922', 'fear'], 'bloodthirsty': ['0.922', 'fear'], 'murderous': ['0.920', 'fear'], 'massacre': ['0.911', 'fear'], 'horrifying': ['0.906', 'fear'], 'mutilation': ['0.906', 'fear'], 'assassinate': ['0.906', 'fear'], 'terrifying': ['0.906', 'fear'], 'fatality': ['0.906', 'fear'], 'horrors': ['0.906', 'fear'], 'demon': ['0.906', 'fear'], 'murder': ['0.906', 'fear'], 'devastation': ['0.906', 'fear'], 'killing': ['0.906', 'fear'], 'terrified': ['0.906', 'fear'], 'holocaust': ['0.906', 'fear'], 'suicidal': ['0.898', 'fear'], 'kidnap': ['0.891', 'fear'], 'crucifixion': ['0.891', 'fear'], 'slaughter': ['0.891', 'fear'], 'assault': ['0.891', 'fear'], 'doomed': ['0.888', 'fear'], 'poisoned': ['0.886', 'fear'], 'suicide': ['0.879', 'fear'], 'explosion': ['0.879', 'fear'], 'disastrous': ['0.875', 'fear'], 'dismemberment': ['0.875', 'fear'], 'annihilation': ['0.875', 'fear'], 'savagery': ['0.875', 'fear'], 'deadly': ['0.875', 'fear'], 'slaughtering': ['0.875', 'fear'], 'suffocation': ['0.875', 'fear'], 'disaster': ['0.875', 'fear'], 'threatening': ['0.875', 'fear'], 'assassin': ['0.875', 'fear'], 'rape': ['0.870', 'fear'], 'hell': ['0.860', 'fear'], 'slaughterhouse': ['0.859', 'fear'], 'guillotine': ['0.859', 'fear'], 'explosive': ['0.859', 'fear'], 'annihilated': ['0.859', 'fear'], 'demonic': ['0.859', 'fear'], 'ihatespiders': ['0.859', 'fear'], 'bloodshed': ['0.859', 'fear'], 'dread': ['0.859', 'fear'], 'homicide': ['0.859', 'fear'], 'barbaric': ['0.859', 'fear'], 'anthrax': ['0.859', 'fear'], 'molestation': ['0.859', 'fear'], 'warfare': ['0.859', 'fear'], 'peril': ['0.859', 'fear'], 'tragedy': ['0.859', 'fear'], 'attacking': ['0.859', 'fear'], 'paralyzed': ['0.859', 'fear'], 'executioner': ['0.859', 'fear'], 'suffocating': ['0.858', 'fear'], 'treachery': ['0.856', 'fear'], 'fright': ['0.853', 'fear'], 'apocalypse': ['0.844', 'fear'], 'bombardment': ['0.844', 'fear'], 'afraid': ['0.844', 'fear'], 'frightening': ['0.844', 'fear'], 'frightened': ['0.844', 'fear'], 'scariest': ['0.844', 'fear'], 'panicked': ['0.844', 'fear'], 'morgue': ['0.844', 'fear'], 'traumatic': ['0.844', 'fear'], 'execution': ['0.844', 'fear'], 'monster': ['0.844', 'fear'], 'vengeance': ['0.844', 'fear'], 'destroying': ['0.844', 'fear'], 'slayer': ['0.844', 'fear'], 'abomination': ['0.844', 'fear'], 'painful': ['0.844', 'fear'], 'drown': ['0.844', 'fear'], 'petrified': ['0.844', 'fear'], 'scare': ['0.844', 'fear'], 'death': ['0.843', 'fear'], 'chaos': ['0.839', 'fear'], 'ghastly': ['0.836', 'fear'], 'evil': ['0.833', 'fear'], 'explode': ['0.828', 'fear'], 'devil': ['0.828', 'fear'], 'fatal': ['0.828', 'fear'], 'doomsday': ['0.828', 'fear'], 'doom': ['0.828', 'fear'], 'frighten': ['0.828', 'fear'], 'cancer': ['0.828', 'fear'], 'fear': ['0.828', 'fear'], 'nightmare': ['0.828', 'fear'], 'manslaughter': ['0.828', 'fear'], 'trauma': ['0.828', 'fear'], 'eradication': ['0.828', 'fear'], 'gunmen': ['0.828', 'fear'], 'intruder': ['0.828', 'fear'], 'brutal': ['0.828', 'fear'], 'violently': ['0.828', 'fear'], 'grenade': ['0.828', 'fear'], 'hellish': ['0.828', 'fear'], 'assassination': ['0.828', 'fear'], 'kidnapped': ['0.828', 'fear'], 'paralyze': ['0.828', 'fear'], 'morbidity': ['0.820', 'fear'], 'crippling': ['0.817', 'fear'], 'savage': ['0.814', 'fear'], 'violence': ['0.812', 'fear'], 'destructive': ['0.812', 'fear'], 'crushed': ['0.812', 'fear'], 'suffering': ['0.812', 'fear'], 'violent': ['0.812', 'fear'], 'heartattack': ['0.812', 'fear'], 'damnation': ['0.812', 'fear'], 'poisonous': ['0.812', 'fear'], 'aggressor': ['0.812', 'fear'], 'earthquake': ['0.812', 'fear'], 'shooting': ['0.812', 'fear'], 'quake': ['0.812', 'fear'], 'frightful': ['0.812', 'fear'], 'hurricane': ['0.811', 'fear'], 'imprisoned': ['0.811', 'fear'], 'exterminate': ['0.810', 'fear'], 'fearing': ['0.808', 'fear'], 'hemorrhage': ['0.807', 'fear'], 'torment': ['0.806', 'fear'], 'lethal': ['0.806', 'fear'], 'venom': ['0.804', 'fear'], 'claustrophobia': ['0.803', 'fear'], 'snakes': ['0.802', 'fear'], 'danger': ['0.802', 'fear'], 'robbery': ['0.800', 'fear'], 'exorcism': ['0.800', 'fear'], 'obliterated': ['0.800', 'fear'], 'terrifies': ['0.798', 'fear'], 'extermination': ['0.797', 'fear'], 'harmful': ['0.797', 'fear'], 'incurable': ['0.797', 'fear'], 'poison': ['0.797', 'fear'], 'cruelty': ['0.797', 'fear'], 'biggestfear': ['0.797', 'fear'], 'mortality': ['0.797', 'fear'], 'monstrosity': ['0.797', 'fear'], 'destruction': ['0.797', 'fear'], 'persecution': ['0.797', 'fear'], 'anxietyattack': ['0.797', 'fear'], 'attack': ['0.797', 'fear'], 'hysteria': ['0.797', 'fear'], 'bombers': ['0.797', 'fear'], 'cyanide': ['0.797', 'fear'], 'dreadfully': ['0.796', 'fear'], 'arson': ['0.794', 'fear'], 'devastate': ['0.792', 'fear'], 'tyrant': ['0.788', 'fear'], 'warcrimes': ['0.785', 'fear'], 'perish': ['0.784', 'fear'], 'feared': ['0.782', 'fear'], 'screaming': ['0.781', 'fear'], 'maniac': ['0.781', 'fear'], 'freakingout': ['0.781', 'fear'], 'bloody': ['0.781', 'fear'], 'venomous': ['0.781', 'fear'], 'rampage': ['0.781', 'fear'], 'enslaved': ['0.781', 'fear'], 'warlike': ['0.781', 'fear'], 'lifeless': ['0.781', 'fear'], 'snake': ['0.776', 'fear'], 'epidemic': ['0.776', 'fear'], 'panicattack': ['0.774', 'fear'], 'dangerously': ['0.766', 'fear'], 'invader': ['0.766', 'fear'], 'crisis': ['0.766', 'fear'], 'detonate': ['0.766', 'fear'], 'hazardous': ['0.766', 'fear'], 'jihad': ['0.766', 'fear'], 'die': ['0.766', 'fear'], 'scary': ['0.766', 'fear'], 'invade': ['0.766', 'fear'], 'reprisal': ['0.766', 'fear'], 'obliterate': ['0.766', 'fear'], 'riot': ['0.766', 'fear'], 'criminal': ['0.766', 'fear'], 'tumor': ['0.766', 'fear'], 'violation': ['0.766', 'fear'], 'fearful': ['0.766', 'fear'], 'inferno': ['0.766', 'fear'], 'ohshit': ['0.766', 'fear'], 'shipwreck': ['0.766', 'fear'], 'leprosy': ['0.766', 'fear'], 'claustrophobic': ['0.765', 'fear'], 'hyperventilating': ['0.762', 'fear'], 'nightmares': ['0.759', 'fear'], 'destroyed': ['0.754', 'fear'], 'ghostly': ['0.754', 'fear'], 'hurricanes': ['0.750', 'fear'], 'tyranny': ['0.750', 'fear'], 'panic': ['0.750', 'fear'], 'ptsd': ['0.750', 'fear'], 'dangerous': ['0.750', 'fear'], 'strangle': ['0.750', 'fear'], 'tragedies': ['0.750', 'fear'], 'wrenching': ['0.750', 'fear'], 'hazard': ['0.750', 'fear'], 'destroyer': ['0.750', 'fear'], 'projectiles': ['0.750', 'fear'], 'cholera': ['0.750', 'fear'], 'obliteration': ['0.750', 'fear'], 'slavery': ['0.750', 'fear'], 'imprisonment': ['0.750', 'fear'], 'agony': ['0.750', 'fear'], 'anaconda': ['0.750', 'fear'], 'anarchist': ['0.750', 'fear'], 'treacherous': ['0.750', 'fear'], 'riotous': ['0.750', 'fear'], 'mortuary': ['0.750', 'fear'], 'dreadful': ['0.750', 'fear'], 'anarchy': ['0.750', 'fear'], 'fears': ['0.750', 'fear'], 'accident': ['0.750', 'fear'], 'malignancy': ['0.742', 'fear'], 'bombard': ['0.740', 'fear'], 'cannibal': ['0.740', 'fear'], 'abominable': ['0.738', 'fear'], 'tyrannical': ['0.734', 'fear'], 'depraved': ['0.734', 'fear'], 'mutiny': ['0.734', 'fear'], 'scared': ['0.734', 'fear'], 'dictatorship': ['0.734', 'fear'], 'beast': ['0.734', 'fear'], 'missile': ['0.734', 'fear'], 'cursed': ['0.734', 'fear'], 'melee': ['0.734', 'fear'], 'threaten': ['0.734', 'fear'], 'hostage': ['0.734', 'fear'], 'diseased': ['0.734', 'fear'], 'gore': ['0.734', 'fear'], 'devilish': ['0.734', 'fear'], 'malignant': ['0.734', 'fear'], 'misery': ['0.734', 'fear'], 'horrible': ['0.734', 'fear'], 'shot': ['0.734', 'fear'], 'bomber': ['0.734', 'fear'], 'hyperventilate': ['0.734', 'fear'], 'crash': ['0.734', 'fear'], 'gun': ['0.734', 'fear'], 'shoot': ['0.734', 'fear'], 'victimized': ['0.734', 'fear'], 'paralysis': ['0.734', 'fear'], 'mafia': ['0.734', 'fear'], 'tornado': ['0.734', 'fear'], 'turmoil': ['0.733', 'fear'], 'combat': ['0.728', 'fear'], 'alligator': ['0.727', 'fear'], 'ruin': ['0.725', 'fear'], 'shooter': ['0.722', 'fear'], 'contagious': ['0.720', 'fear'], 'miscarriage': ['0.719', 'fear'], 'lynch': ['0.719', 'fear'], 'desperation': ['0.719', 'fear'], 'casualty': ['0.719', 'fear'], 'devastating': ['0.719', 'fear'], 'seizure': ['0.719', 'fear'], 'starvation': ['0.719', 'fear'], 'excruciating': ['0.719', 'fear'], 'phobia': ['0.719', 'fear'], 'harm': ['0.719', 'fear'], 'crime': ['0.719', 'fear'], 'emergency': ['0.719', 'fear'], 'shock': ['0.719', 'fear'], 'fight': ['0.719', 'fear'], 'injured': ['0.719', 'fear'], 'scream': ['0.719', 'fear'], 'struggle': ['0.719', 'fear'], 'havoc': ['0.719', 'fear'], 'mortification': ['0.719', 'fear'], 'frantically': ['0.717', 'fear'], 'carnage': ['0.717', 'fear'], 'standoff': ['0.716', 'fear'], 'infestation': ['0.716', 'fear'], 'soscary': ['0.712', 'fear'], 'panicking': ['0.708', 'fear'], 'rupture': ['0.706', 'fear'], 'frantic': ['0.705', 'fear'], 'horrid': ['0.705', 'fear'], 'burial': ['0.703', 'fear'], 'arsenic': ['0.703', 'fear'], 'neurotic': ['0.703', 'fear'], 'carcinoma': ['0.703', 'fear'], 'witchcraft': ['0.703', 'fear'], 'hysterical': ['0.703', 'fear'], 'deranged': ['0.703', 'fear'], 'prey': ['0.703', 'fear'], 'infectious': ['0.703', 'fear'], 'assailant': ['0.703', 'fear'], 'sabotage': ['0.703', 'fear'], 'psychosis': ['0.703', 'fear'], 'hatred': ['0.703', 'fear'], 'terminal': ['0.703', 'fear'], 'collapse': ['0.703', 'fear'], 'anguish': ['0.703', 'fear'], 'grisly': ['0.703', 'fear'], 'diabolical': ['0.703', 'fear'], 'avalanche': ['0.703', 'fear'], 'dreaded': ['0.703', 'fear'], 'gash': ['0.703', 'fear'], 'sos': ['0.703', 'fear'], 'aggressive': ['0.703', 'fear'], 'malevolent': ['0.703', 'fear'], 'agonizing': ['0.703', 'fear'], 'cobra': ['0.703', 'fear'], 'harrowing': ['0.703', 'fear'], 'fugitive': ['0.703', 'fear'], 'wrecked': ['0.703', 'fear'], 'armed': ['0.703', 'fear'], 'cripple': ['0.703', 'fear'], 'freaked': ['0.703', 'fear'], 'schizophrenia': ['0.703', 'fear'], 'plague': ['0.703', 'fear'], 'upheaval': ['0.703', 'fear'], 'vendetta': ['0.703', 'fear'], 'forcibly': ['0.700', 'fear'], 'abduction': ['0.700', 'fear'], 'crocodile': ['0.700', 'fear'], 'offender': ['0.698', 'fear'], 'mayhem': ['0.690', 'fear'], 'wounding': ['0.688', 'fear'], 'disease': ['0.688', 'fear'], 'agoraphobia': ['0.688', 'fear'], 'brawl': ['0.688', 'fear'], 'banish': ['0.688', 'fear'], 'oppression': ['0.688', 'fear'], 'hostile': ['0.688', 'fear'], 'guerilla': ['0.688', 'fear'], 'cyclone': ['0.688', 'fear'], 'radiation': ['0.688', 'fear'], 'hopeless': ['0.688', 'fear'], 'ambush': ['0.688', 'fear'], 'beastly': ['0.688', 'fear'], 'shrapnel': ['0.688', 'fear'], 'socialanxiety': ['0.688', 'fear'], 'cadaver': ['0.688', 'fear'], 'leukemia': ['0.688', 'fear'], 'rabid': ['0.688', 'fear'], 'endanger': ['0.688', 'fear'], 'contagion': ['0.688', 'fear'], 'anarchism': ['0.688', 'fear'], 'hurt': ['0.688', 'fear'], 'trepidation': ['0.688', 'fear'], 'gory': ['0.688', 'fear'], 'paranoid': ['0.688', 'fear'], 'alarm': ['0.688', 'fear'], 'blast': ['0.688', 'fear'], 'gallows': ['0.686', 'fear'], 'coffin': ['0.684', 'fear'], 'madness': ['0.675', 'fear'], 'spook': ['0.673', 'fear'], 'fury': ['0.672', 'fear'], 'awful': ['0.672', 'fear'], 'victim': ['0.672', 'fear'], 'ghost': ['0.672', 'fear'], 'trembling': ['0.672', 'fear'], 'scarier': ['0.672', 'fear'], 'stab': ['0.672', 'fear'], 'viper': ['0.672', 'fear'], 'nervouswreck': ['0.672', 'fear'], 'tarantula': ['0.672', 'fear'], 'thug': ['0.672', 'fear'], 'dungeon': ['0.672', 'fear'], 'carcass': ['0.672', 'fear'], 'freakedout': ['0.672', 'fear'], 'vermin': ['0.672', 'fear'], 'felon': ['0.672', 'fear'], 'atrocity': ['0.672', 'fear'], 'fangs': ['0.672', 'fear'], 'militants': ['0.672', 'fear'], 'perishing': ['0.672', 'fear'], 'militia': ['0.672', 'fear'], 'grim': ['0.672', 'fear'], 'outbreak': ['0.672', 'fear'], 'abhorrent': ['0.672', 'fear'], 'abuse': ['0.672', 'fear'], 'frankenstorm': ['0.672', 'fear'], 'sinister': ['0.672', 'fear'], 'menace': ['0.672', 'fear'], 'hurting': ['0.672', 'fear'], 'gang': ['0.672', 'fear'], 'crushing': ['0.672', 'fear'], 'combatant': ['0.672', 'fear'], 'eruption': ['0.672', 'fear'], 'shatter': ['0.672', 'fear'], 'persecute': ['0.672', 'fear'], 'revolver': ['0.672', 'fear'], 'injury': ['0.672', 'fear'], 'mangle': ['0.672', 'fear'], 'soscared': ['0.672', 'fear'], 'menacing': ['0.672', 'fear'], 'desolation': ['0.667', 'fear'], 'ferocious': ['0.667', 'fear'], 'battered': ['0.667', 'fear'], 'cutthroat': ['0.664', 'fear'], 'pandemic': ['0.664', 'fear'], 'volcano': ['0.663', 'fear'], 'scares': ['0.660', 'fear'], 'cruelly': ['0.658', 'fear'], 'nefarious': ['0.656', 'fear'], 'fearfully': ['0.656', 'fear'], 'purgatory': ['0.656', 'fear'], 'despotic': ['0.656', 'fear'], 'tumult': ['0.656', 'fear'], 'malicious': ['0.656', 'fear'], 'emetophobia': ['0.656', 'fear'], 'selfharm': ['0.656', 'fear'], 'lunatic': ['0.656', 'fear'], 'bully': ['0.656', 'fear'], 'typhoon': ['0.656', 'fear'], 'xenophobia': ['0.656', 'fear'], 'treason': ['0.656', 'fear'], 'oppressive': ['0.656', 'fear'], 'battlefield': ['0.656', 'fear'], 'dictator': ['0.656', 'fear'], 'smash': ['0.656', 'fear'], 'radioactive': ['0.656', 'fear'], 'encroachment': ['0.656', 'fear'], 'stroke': ['0.656', 'fear'], 'traitor': ['0.656', 'fear'], 'blizzard': ['0.656', 'fear'], 'coma': ['0.656', 'fear'], 'cruel': ['0.656', 'fear'], 'endangered': ['0.656', 'fear'], 'armament': ['0.656', 'fear'], 'ruthless': ['0.656', 'fear'], 'punishment': ['0.656', 'fear'], 'shitless': ['0.656', 'fear'], 'neurosis': ['0.656', 'fear'], 'debauchery': ['0.656', 'fear'], 'spider': ['0.656', 'fear'], 'distress': ['0.656', 'fear'], 'alarming': ['0.656', 'fear'], 'revenge': ['0.656', 'fear'], 'crazed': ['0.656', 'fear'], 'insane': ['0.656', 'fear'], 'projectile': ['0.654', 'fear'], 'flog': ['0.653', 'fear'], 'bang': ['0.652', 'fear'], 'ransom': ['0.644', 'fear'], 'criminality': ['0.642', 'fear'], 'hanging': ['0.641', 'fear'], 'deteriorate': ['0.641', 'fear'], 'dementia': ['0.641', 'fear'], 'punished': ['0.641', 'fear'], 'perturbation': ['0.641', 'fear'], 'haunt': ['0.641', 'fear'], 'raptors': ['0.641', 'fear'], 'abortion': ['0.641', 'fear'], 'despair': ['0.641', 'fear'], 'cantbreathe': ['0.641', 'fear'], 'scoundrel': ['0.641', 'fear'], 'isolated': ['0.641', 'fear'], 'quivering': ['0.641', 'fear'], 'duress': ['0.641', 'fear'], 'dreading': ['0.641', 'fear'], 'targeted': ['0.641', 'fear'], 'rattlesnake': ['0.641', 'fear'], 'demise': ['0.641', 'fear'], 'warlock': ['0.641', 'fear'], 'perilous': ['0.641', 'fear'], 'dastardly': ['0.641', 'fear'], 'raging': ['0.641', 'fear'], 'injurious': ['0.641', 'fear'], 'intimidate': ['0.641', 'fear'], 'conflict': ['0.641', 'fear'], 'wildfire': ['0.641', 'fear'], 'wreck': ['0.641', 'fear'], 'tumour': ['0.641', 'fear'], 'retaliation': ['0.641', 'fear'], 'vehement': ['0.641', 'fear'], 'demented': ['0.641', 'fear'], 'hideous': ['0.641', 'fear'], 'banshee': ['0.641', 'fear'], 'antichrist': ['0.641', 'fear'], 'aghast': ['0.641', 'fear'], 'bestial': ['0.639', 'fear'], 'serpent': ['0.638', 'fear'], 'condemnation': ['0.637', 'fear'], 'corrosive': ['0.636', 'fear'], 'armaments': ['0.636', 'fear'], 'fire': ['0.636', 'fear'], 'exclusion': ['0.636', 'fear'], 'nervousness': ['0.627', 'fear'], 'seize': ['0.625', 'fear'], 'pneumonia': ['0.625', 'fear'], 'spiders': ['0.625', 'fear'], 'revolting': ['0.625', 'fear'], 'vampire': ['0.625', 'fear'], 'mercenary': ['0.625', 'fear'], 'sarcoma': ['0.625', 'fear'], 'incarceration': ['0.625', 'fear'], 'manic': ['0.625', 'fear'], 'injure': ['0.625', 'fear'], 'smuggler': ['0.625', 'fear'], 'pestilence': ['0.625', 'fear'], 'hurtful': ['0.625', 'fear'], 'stunned': ['0.625', 'fear'], 'vengeful': ['0.625', 'fear'], 'irreparable': ['0.625', 'fear'], 'lunacy': ['0.625', 'fear'], 'thundering': ['0.625', 'fear'], 'disfigured': ['0.625', 'fear'], 'artillery': ['0.625', 'fear'], 'incendiary': ['0.625', 'fear'], 'daemon': ['0.625', 'fear'], 'wracking': ['0.625', 'fear'], 'turbulent': ['0.625', 'fear'], 'landslide': ['0.625', 'fear'], 'malice': ['0.625', 'fear'], 'enemy': ['0.625', 'fear'], 'combative': ['0.625', 'fear'], 'sickening': ['0.625', 'fear'], 'repression': ['0.625', 'fear'], 'infection': ['0.625', 'fear'], 'wicked': ['0.625', 'fear'], 'inhuman': ['0.625', 'fear'], 'meltdown': ['0.625', 'fear'], 'prison': ['0.625', 'fear'], 'ruinous': ['0.625', 'fear'], 'madman': ['0.625', 'fear'], 'virulence': ['0.625', 'fear'], 'ill': ['0.621', 'fear'], 'barbarian': ['0.620', 'fear'], 'battled': ['0.615', 'fear'], 'plummet': ['0.613', 'fear'], 'blackmail': ['0.612', 'fear'], 'hangman': ['0.610', 'fear'], 'contaminated': ['0.610', 'fear'], 'darkened': ['0.610', 'fear'], 'merciless': ['0.609', 'fear'], 'beating': ['0.609', 'fear'], 'aggression': ['0.609', 'fear'], 'injection': ['0.609', 'fear'], 'alienation': ['0.609', 'fear'], 'insurmountable': ['0.609', 'fear'], 'dagger': ['0.609', 'fear'], 'conflagration': ['0.609', 'fear'], 'butcher': ['0.609', 'fear'], 'malaria': ['0.609', 'fear'], 'interrogation': ['0.609', 'fear'], 'despotism': ['0.609', 'fear'], 'wrath': ['0.609', 'fear'], 'prosecute': ['0.609', 'fear'], 'disintegrate': ['0.609', 'fear'], 'shackle': ['0.609', 'fear'], 'oblivion': ['0.609', 'fear'], 'homeless': ['0.609', 'fear'], 'sufferer': ['0.609', 'fear'], 'rob': ['0.609', 'fear'], 'ogre': ['0.609', 'fear'], 'oppressor': ['0.609', 'fear'], 'abandonment': ['0.609', 'fear'], 'deplorable': ['0.609', 'fear'], 'injuring': ['0.609', 'fear'], 'illness': ['0.609', 'fear'], 'bleeding': ['0.609', 'fear'], 'infanticide': ['0.609', 'fear'], 'scourge': ['0.609', 'fear'], 'threat': ['0.604', 'fear'], 'firearms': ['0.600', 'fear'], 'nerves': ['0.600', 'fear'], 'raid': ['0.600', 'fear'], 'eviction': ['0.596', 'fear'], 'villain': ['0.595', 'fear'], 'torrent': ['0.594', 'fear'], 'pounding': ['0.594', 'fear'], 'haze': ['0.594', 'fear'], 'cardiomyopathy': ['0.594', 'fear'], 'infidel': ['0.594', 'fear'], 'deceit': ['0.594', 'fear'], 'possessed': ['0.594', 'fear'], 'dragon': ['0.594', 'fear'], 'terrible': ['0.594', 'fear'], 'belligerent': ['0.594', 'fear'], 'restrained': ['0.594', 'fear'], 'goblin': ['0.594', 'fear'], 'hypertrophy': ['0.594', 'fear'], 'paranoia': ['0.594', 'fear'], 'tribulation': ['0.594', 'fear'], 'flee': ['0.594', 'fear'], 'shelling': ['0.594', 'fear'], 'masochism': ['0.594', 'fear'], 'tremor': ['0.594', 'fear'], 'burglar': ['0.594', 'fear'], 'sickness': ['0.594', 'fear'], 'anxiety': ['0.594', 'fear'], 'pain': ['0.594', 'fear'], 'sepsis': ['0.594', 'fear'], 'euthanasia': ['0.594', 'fear'], 'omen': ['0.594', 'fear'], 'foe': ['0.594', 'fear'], 'ominous': ['0.594', 'fear'], 'fraught': ['0.594', 'fear'], 'convict': ['0.594', 'fear'], 'rot': ['0.594', 'fear'], 'casket': ['0.594', 'fear'], 'accursed': ['0.594', 'fear'], 'lightning': ['0.594', 'fear'], 'villainous': ['0.594', 'fear'], 'wounded': ['0.592', 'fear'], 'elimination': ['0.588', 'fear'], 'lava': ['0.588', 'fear'], 'spank': ['0.587', 'fear'], 'hostilities': ['0.586', 'fear'], 'dismal': ['0.584', 'fear'], 'blockade': ['0.582', 'fear'], 'punch': ['0.580', 'fear'], 'haunted': ['0.578', 'fear'], 'angina': ['0.578', 'fear'], 'infliction': ['0.578', 'fear'], 'dispossessed': ['0.578', 'fear'], 'tumultuous': ['0.578', 'fear'], 'quarantine': ['0.578', 'fear'], 'desecration': ['0.578', 'fear'], 'coward': ['0.578', 'fear'], 'eatingdisorders': ['0.578', 'fear'], 'grizzly': ['0.578', 'fear'], 'demoralized': ['0.578', 'fear'], 'worry': ['0.578', 'fear'], 'curse': ['0.578', 'fear'], 'deprivation': ['0.578', 'fear'], 'shudder': ['0.578', 'fear'], 'bacteria': ['0.578', 'fear'], 'pained': ['0.578', 'fear'], 'stealing': ['0.578', 'fear'], 'failure': ['0.578', 'fear'], 'sostressed': ['0.578', 'fear'], 'hateful': ['0.578', 'fear'], 'forbidding': ['0.578', 'fear'], 'cringe': ['0.578', 'fear'], 'parasite': ['0.578', 'fear'], 'loss': ['0.578', 'fear'], 'abyss': ['0.578', 'fear'], 'surgery': ['0.578', 'fear'], 'volatility': ['0.578', 'fear'], 'apprehensive': ['0.578', 'fear'], 'confined': ['0.578', 'fear'], 'clashing': ['0.578', 'fear'], 'jeopardy': ['0.578', 'fear'], 'shady': ['0.578', 'fear'], 'jail': ['0.578', 'fear'], 'dire': ['0.578', 'fear'], 'evacuate': ['0.578', 'fear'], 'assail': ['0.578', 'fear'], 'perdition': ['0.577', 'fear'], 'mob': ['0.577', 'fear'], 'toxin': ['0.575', 'fear'], 'unholy': ['0.575', 'fear'], 'comatose': ['0.575', 'fear'], 'pillage': ['0.574', 'fear'], 'incest': ['0.571', 'fear'], 'wound': ['0.571', 'fear'], 'forced': ['0.569', 'fear'], 'cowardice': ['0.567', 'fear'], 'domination': ['0.566', 'fear'], 'crypt': ['0.566', 'fear'], 'witch': ['0.565', 'fear'], 'smuggle': ['0.565', 'fear'], 'worries': ['0.562', 'fear'], 'banished': ['0.562', 'fear'], 'purge': ['0.562', 'fear'], 'inflict': ['0.562', 'fear'], 'masks': ['0.562', 'fear'], 'afflict': ['0.562', 'fear'], 'sacrifices': ['0.562', 'fear'], 'deterioration': ['0.562', 'fear'], 'eeek': ['0.562', 'fear'], 'gunpowder': ['0.562', 'fear'], 'misfortune': ['0.562', 'fear'], 'ulcer': ['0.562', 'fear'], 'incubus': ['0.562', 'fear'], 'apprehend': ['0.562', 'fear'], 'lawsuit': ['0.562', 'fear'], 'distressing': ['0.562', 'fear'], 'prisoner': ['0.562', 'fear'], 'jitters': ['0.562', 'fear'], 'corrupting': ['0.562', 'fear'], 'thrash': ['0.562', 'fear'], 'martyr': ['0.562', 'fear'], 'travesty': ['0.562', 'fear'], 'duel': ['0.562', 'fear'], 'pitfall': ['0.562', 'fear'], 'evasion': ['0.562', 'fear'], 'overpowering': ['0.562', 'fear'], 'vanished': ['0.562', 'fear'], 'impending': ['0.562', 'fear'], 'risky': ['0.562', 'fear'], 'martyrdom': ['0.562', 'fear'], 'appalling': ['0.562', 'fear'], 'syncope': ['0.562', 'fear'], 'polio': ['0.562', 'fear'], 'risk': ['0.562', 'fear'], 'confine': ['0.562', 'fear'], 'grave': ['0.562', 'fear'], 'turbulence': ['0.562', 'fear'], 'degrading': ['0.562', 'fear'], 'punishing': ['0.562', 'fear'], 'powerless': ['0.562', 'fear'], 'incrimination': ['0.562', 'fear'], 'freak': ['0.562', 'fear'], 'mange': ['0.562', 'fear'], 'hunter': ['0.562', 'fear'], 'unsafe': ['0.561', 'fear'], 'sin': ['0.560', 'fear'], 'worstfeeling': ['0.560', 'fear'], 'perpetrator': ['0.560', 'fear'], 'occult': ['0.559', 'fear'], 'intimidation': ['0.559', 'fear'], 'disable': ['0.558', 'fear'], 'decay': ['0.557', 'fear'], 'affliction': ['0.557', 'fear'], 'autopsy': ['0.557', 'fear'], 'vulnerability': ['0.548', 'fear'], 'flood': ['0.547', 'fear'], 'bondage': ['0.547', 'fear'], 'abhor': ['0.547', 'fear'], 'overthrow': ['0.547', 'fear'], 'appendicitis': ['0.547', 'fear'], 'outcry': ['0.547', 'fear'], 'cannon': ['0.547', 'fear'], 'siren': ['0.547', 'fear'], 'growling': ['0.547', 'fear'], 'jeopardize': ['0.547', 'fear'], 'deformity': ['0.547', 'fear'], 'samurai': ['0.547', 'fear'], 'neuralgia': ['0.547', 'fear'], 'talons': ['0.547', 'fear'], 'defenseless': ['0.547', 'fear'], 'beware': ['0.547', 'fear'], 'noxious': ['0.547', 'fear'], 'asylum': ['0.547', 'fear'], 'scorpion': ['0.547', 'fear'], 'outburst': ['0.547', 'fear'], 'derogation': ['0.547', 'fear'], 'enmity': ['0.547', 'fear'], 'ailing': ['0.547', 'fear'], 'scandal': ['0.547', 'fear'], 'adder': ['0.547', 'fear'], 'communism': ['0.547', 'fear'], 'thief': ['0.547', 'fear'], 'plunge': ['0.547', 'fear'], 'precarious': ['0.547', 'fear'], 'expulsion': ['0.547', 'fear'], 'mad': ['0.547', 'fear'], 'brigade': ['0.547', 'fear'], 'needles': ['0.547', 'fear'], 'insanity': ['0.547', 'fear'], 'apparition': ['0.547', 'fear'], 'senile': ['0.547', 'fear'], 'coercion': ['0.547', 'fear'], 'gonorrhea': ['0.547', 'fear'], 'retribution': ['0.545', 'fear'], 'uprising': ['0.545', 'fear'], 'cemetery': ['0.541', 'fear'], 'badness': ['0.539', 'fear'], 'phantom': ['0.538', 'fear'], 'abandoned': ['0.534', 'fear'], 'fled': ['0.534', 'fear'], 'rejection': ['0.533', 'fear'], 'rebellion': ['0.531', 'fear'], 'disembodied': ['0.531', 'fear'], 'captor': ['0.531', 'fear'], 'stormy': ['0.531', 'fear'], 'stalk': ['0.531', 'fear'], 'wrangling': ['0.531', 'fear'], 'helpless': ['0.531', 'fear'], 'subjugation': ['0.531', 'fear'], 'slave': ['0.531', 'fear'], 'hopelessness': ['0.531', 'fear'], 'superstitious': ['0.531', 'fear'], 'broken': ['0.531', 'fear'], 'manifestation': ['0.531', 'fear'], 'cowardly': ['0.531', 'fear'], 'abandon': ['0.531', 'fear'], 'deserted': ['0.531', 'fear'], 'jarring': ['0.531', 'fear'], 'frenzied': ['0.531', 'fear'], 'theft': ['0.531', 'fear'], 'bacterium': ['0.531', 'fear'], 'saber': ['0.531', 'fear'], 'manipulation': ['0.531', 'fear'], 'lawlessness': ['0.531', 'fear'], 'impotence': ['0.531', 'fear'], 'penetration': ['0.531', 'fear'], 'swastika': ['0.531', 'fear'], 'atherosclerosis': ['0.531', 'fear'], 'shaking': ['0.531', 'fear'], 'carnivorous': ['0.531', 'fear'], 'eek': ['0.531', 'fear'], 'leeches': ['0.531', 'fear'], 'buried': ['0.531', 'fear'], 'harbinger': ['0.531', 'fear'], 'endocarditis': ['0.531', 'fear'], 'failing': ['0.531', 'fear'], 'stressed': ['0.531', 'fear'], 'punish': ['0.531', 'fear'], 'unstable': ['0.531', 'fear'], 'freakout': ['0.531', 'fear'], 'incursion': ['0.531', 'fear'], 'distrust': ['0.531', 'fear'], 'suspense': ['0.529', 'fear'], 'blood': ['0.525', 'fear'], 'bear': ['0.524', 'fear'], 'hiding': ['0.524', 'fear'], 'unlawful': ['0.519', 'fear'], 'crazy': ['0.519', 'fear'], 'anxious': ['0.518', 'fear'], 'cult': ['0.518', 'fear'], 'captive': ['0.517', 'fear'], 'deportation': ['0.517', 'fear'], 'detainee': ['0.516', 'fear'], 'enforce': ['0.516', 'fear'], 'suspect': ['0.516', 'fear'], 'oppress': ['0.516', 'fear'], 'revulsion': ['0.516', 'fear'], 'grievous': ['0.516', 'fear'], 'fang': ['0.516', 'fear'], 'exile': ['0.516', 'fear'], 'insidious': ['0.516', 'fear'], 'plunder': ['0.516', 'fear'], 'foreboding': ['0.516', 'fear'], 'intolerance': ['0.516', 'fear'], 'incite': ['0.516', 'fear'], 'sorcery': ['0.516', 'fear'], 'emaciated': ['0.516', 'fear'], 'shriek': ['0.516', 'fear'], 'busted': ['0.516', 'fear'], 'helplessness': ['0.516', 'fear'], 'exacerbation': ['0.516', 'fear'], 'biopsy': ['0.516', 'fear'], 'badfeeling': ['0.516', 'fear'], 'relapse': ['0.516', 'fear'], 'plight': ['0.516', 'fear'], 'growl': ['0.516', 'fear'], 'troublesome': ['0.516', 'fear'], 'chased': ['0.516', 'fear'], 'embolism': ['0.516', 'fear'], 'darkness': ['0.516', 'fear'], 'conspirator': ['0.516', 'fear'], 'palsy': ['0.516', 'fear'], 'harshness': ['0.516', 'fear'], 'unthinkable': ['0.516', 'fear'], 'sting': ['0.516', 'fear'], 'robber': ['0.516', 'fear'], 'pernicious': ['0.516', 'fear'], 'smite': ['0.516', 'fear'], 'depression': ['0.508', 'fear'], 'confinement': ['0.507', 'fear'], 'snowmageddon': ['0.500', 'fear'], 'illegal': ['0.500', 'fear'], 'eeeek': ['0.500', 'fear'], 'towering': ['0.500', 'fear'], 'intolerant': ['0.500', 'fear'], 'subvert': ['0.500', 'fear'], 'cliff': ['0.500', 'fear'], 'sinful': ['0.500', 'fear'], 'wasp': ['0.500', 'fear'], 'suppress': ['0.500', 'fear'], 'brute': ['0.500', 'fear'], 'prohibited': ['0.500', 'fear'], 'suppression': ['0.500', 'fear'], 'ruined': ['0.500', 'fear'], 'instability': ['0.500', 'fear'], 'constraint': ['0.500', 'fear'], 'odious': ['0.500', 'fear'], 'downfall': ['0.500', 'fear'], 'surrender': ['0.500', 'fear'], 'raving': ['0.500', 'fear'], 'mistrust': ['0.500', 'fear'], 'alcoholism': ['0.500', 'fear'], 'blob': ['0.500', 'fear'], 'frenetic': ['0.500', 'fear'], 'scalpel': ['0.500', 'fear'], 'vertigo': ['0.500', 'fear'], 'prayforme': ['0.500', 'fear'], 'infarct': ['0.500', 'fear'], 'defiance': ['0.500', 'fear'], 'squeamish': ['0.500', 'fear'], 'madden': ['0.500', 'fear'], 'premeditated': ['0.500', 'fear'], 'hunting': ['0.500', 'fear'], 'woe': ['0.491', 'fear'], 'intrusion': ['0.484', 'fear'], 'hearse': ['0.484', 'fear'], 'lion': ['0.484', 'fear'], 'armor': ['0.484', 'fear'], 'hate': ['0.484', 'fear'], 'rat': ['0.484', 'fear'], 'syringe': ['0.484', 'fear'], 'ravine': ['0.484', 'fear'], 'endemic': ['0.484', 'fear'], 'blackness': ['0.484', 'fear'], 'divorce': ['0.484', 'fear'], 'disgusting': ['0.484', 'fear'], 'hallucination': ['0.484', 'fear'], 'defamation': ['0.484', 'fear'], 'rabble': ['0.484', 'fear'], 'ambulance': ['0.484', 'fear'], 'hospital': ['0.484', 'fear'], 'worse': ['0.484', 'fear'], 'dismay': ['0.484', 'fear'], 'indictment': ['0.484', 'fear'], 'dishonor': ['0.484', 'fear'], 'poaching': ['0.484', 'fear'], 'freakish': ['0.484', 'fear'], 'cutting': ['0.484', 'fear'], 'jumpy': ['0.484', 'fear'], 'insecurity': ['0.484', 'fear'], 'perjury': ['0.484', 'fear'], 'lash': ['0.484', 'fear'], 'accusing': ['0.484', 'fear'], 'slam': ['0.484', 'fear'], 'ravenous': ['0.484', 'fear'], 'sonervous': ['0.484', 'fear'], 'fiend': ['0.484', 'fear'], 'palpitations': ['0.484', 'fear'], 'coerce': ['0.484', 'fear'], 'canthandleit': ['0.484', 'fear'], 'retard': ['0.484', 'fear'], 'cracked': ['0.484', 'fear'], 'restriction': ['0.484', 'fear'], 'indefensible': ['0.484', 'fear'], 'prejudiced': ['0.484', 'fear'], 'shiver': ['0.484', 'fear'], 'warrior': ['0.484', 'fear'], 'worrying': ['0.484', 'fear'], 'flinch': ['0.484', 'fear'], 'ohno': ['0.484', 'fear'], 'sinner': ['0.483', 'fear'], 'averse': ['0.483', 'fear'], 'inspection': ['0.483', 'fear'], 'despairing': ['0.474', 'fear'], 'missing': ['0.474', 'fear'], 'darken': ['0.471', 'fear'], 'austere': ['0.470', 'fear'], 'depressed': ['0.469', 'fear'], 'quiver': ['0.469', 'fear'], 'overwhelmed': ['0.469', 'fear'], 'unhealthy': ['0.469', 'fear'], 'decomposition': ['0.469', 'fear'], 'toothache': ['0.469', 'fear'], 'foreclose': ['0.469', 'fear'], 'infidelity': ['0.469', 'fear'], 'superstition': ['0.469', 'fear'], 'subversion': ['0.469', 'fear'], 'illegality': ['0.469', 'fear'], 'animosity': ['0.469', 'fear'], 'inmate': ['0.469', 'fear'], 'criticize': ['0.469', 'fear'], 'revoke': ['0.469', 'fear'], 'accused': ['0.469', 'fear'], 'sinking': ['0.469', 'fear'], 'intrusive': ['0.469', 'fear'], 'soulless': ['0.469', 'fear'], 'thresh': ['0.469', 'fear'], 'aftermath': ['0.469', 'fear'], 'scrapie': ['0.469', 'fear'], 'fierce': ['0.469', 'fear'], 'fret': ['0.469', 'fear'], 'nervous': ['0.469', 'fear'], 'breakdown': ['0.469', 'fear'], 'jaws': ['0.469', 'fear'], 'idolatry': ['0.469', 'fear'], 'asteroid': ['0.469', 'fear'], 'fever': ['0.469', 'fear'], 'admonition': ['0.469', 'fear'], 'precipice': ['0.469', 'fear'], 'regime': ['0.469', 'fear'], 'armored': ['0.469', 'fear'], 'indict': ['0.469', 'fear'], 'startle': ['0.469', 'fear'], 'spear': ['0.468', 'fear'], 'worried': ['0.466', 'fear'], 'nasty': ['0.466', 'fear'], 'thirteenth': ['0.465', 'fear'], 'puma': ['0.465', 'fear'], 'contraband': ['0.464', 'fear'], 'insecure': ['0.461', 'fear'], 'disturbance': ['0.459', 'fear'], 'denunciation': ['0.456', 'fear'], 'destitute': ['0.455', 'fear'], 'stigma': ['0.455', 'fear'], 'powerfully': ['0.454', 'fear'], 'derogatory': ['0.453', 'fear'], 'batter': ['0.453', 'fear'], 'warn': ['0.453', 'fear'], 'rifle': ['0.453', 'fear'], 'resign': ['0.453', 'fear'], 'quandary': ['0.453', 'fear'], 'scold': ['0.453', 'fear'], 'straits': ['0.453', 'fear'], 'disreputable': ['0.453', 'fear'], 'spasm': ['0.453', 'fear'], 'dinosaur': ['0.453', 'fear'], 'restrain': ['0.453', 'fear'], 'rigor': ['0.453', 'fear'], 'shrill': ['0.453', 'fear'], 'stranger': ['0.453', 'fear'], 'disruption': ['0.453', 'fear'], 'apprehension': ['0.453', 'fear'], 'grope': ['0.453', 'fear'], 'wail': ['0.453', 'fear'], 'unspeakable': ['0.453', 'fear'], 'howl': ['0.453', 'fear'], 'hydrocephalus': ['0.453', 'fear'], 'penalty': ['0.453', 'fear'], 'omgomgomg': ['0.453', 'fear'], 'lockup': ['0.453', 'fear'], 'indoctrination': ['0.453', 'fear'], 'ohgod': ['0.453', 'fear'], 'penal': ['0.453', 'fear'], 'orphan': ['0.453', 'fear'], 'resisting': ['0.453', 'fear'], 'illicit': ['0.453', 'fear'], 'urgent': ['0.450', 'fear'], 'arraignment': ['0.450', 'fear'], 'tramp': ['0.440', 'fear'], 'behemoth': ['0.439', 'fear'], 'stress': ['0.439', 'fear'], 'mace': ['0.438', 'fear'], 'obstruct': ['0.438', 'fear'], 'entangled': ['0.438', 'fear'], 'expel': ['0.438', 'fear'], 'perverse': ['0.438', 'fear'], 'targeting': ['0.438', 'fear'], 'dislocated': ['0.438', 'fear'], 'accuser': ['0.438', 'fear'], 'dentists': ['0.438', 'fear'], 'yell': ['0.438', 'fear'], 'asp': ['0.438', 'fear'], 'forfeiture': ['0.438', 'fear'], 'infirmity': ['0.438', 'fear'], 'dominate': ['0.438', 'fear'], 'tribunal': ['0.438', 'fear'], 'aghhh': ['0.438', 'fear'], 'predicament': ['0.438', 'fear'], 'bewildered': ['0.438', 'fear'], 'looming': ['0.438', 'fear'], 'debacle': ['0.438', 'fear'], 'stripped': ['0.438', 'fear'], 'indomitable': ['0.438', 'fear'], 'unprepared': ['0.438', 'fear'], 'atrophy': ['0.438', 'fear'], 'coldsweat': ['0.438', 'fear'], 'rheumatism': ['0.438', 'fear'], 'falter': ['0.438', 'fear'], 'irreconcilable': ['0.438', 'fear'], 'broke': ['0.438', 'fear'], 'scarcity': ['0.438', 'fear'], 'impermeable': ['0.438', 'fear'], 'cabal': ['0.438', 'fear'], 'probation': ['0.438', 'fear'], 'wince': ['0.438', 'fear'], 'accidental': ['0.438', 'fear'], 'illegitimate': ['0.438', 'fear'], 'gasping': ['0.438', 'fear'], 'concealed': ['0.438', 'fear'], 'displaced': ['0.438', 'fear'], 'prick': ['0.438', 'fear'], 'rebels': ['0.435', 'fear'], 'insulting': ['0.435', 'fear'], 'halting': ['0.435', 'fear'], 'mysterious': ['0.435', 'fear'], 'repulsion': ['0.431', 'fear'], 'glare': ['0.430', 'fear'], 'punitive': ['0.425', 'fear'], 'sordid': ['0.425', 'fear'], 'tempest': ['0.423', 'fear'], 'stake': ['0.423', 'fear'], 'ultimatum': ['0.422', 'fear'], 'infamous': ['0.422', 'fear'], 'reticent': ['0.422', 'fear'], 'dissident': ['0.422', 'fear'], 'coyote': ['0.422', 'fear'], 'deleterious': ['0.422', 'fear'], 'usurped': ['0.422', 'fear'], 'prowl': ['0.422', 'fear'], 'alien': ['0.422', 'fear'], 'depress': ['0.422', 'fear'], 'bier': ['0.422', 'fear'], 'disappear': ['0.422', 'fear'], 'delusion': ['0.422', 'fear'], 'heathen': ['0.422', 'fear'], 'anathema': ['0.422', 'fear'], 'depresson': ['0.422', 'fear'], 'unnatural': ['0.422', 'fear'], 'suspension': ['0.422', 'fear'], 'stresses': ['0.422', 'fear'], 'troll': ['0.422', 'fear'], 'wilderness': ['0.422', 'fear'], 'burdensome': ['0.422', 'fear'], 'offense': ['0.422', 'fear'], 'hydra': ['0.422', 'fear'], 'publicspeaking': ['0.422', 'fear'], 'bayonet': ['0.422', 'fear'], 'varicella': ['0.422', 'fear'], 'lament': ['0.422', 'fear'], 'extrajudicial': ['0.422', 'fear'], 'bigot': ['0.422', 'fear'], 'steal': ['0.422', 'fear'], 'stressin': ['0.422', 'fear'], 'unsteady': ['0.422', 'fear'], 'thumping': ['0.422', 'fear'], 'fainting': ['0.422', 'fear'], 'adversity': ['0.418', 'fear'], 'ineptitude': ['0.417', 'fear'], 'outsider': ['0.413', 'fear'], 'ugliness': ['0.410', 'fear'], 'vulture': ['0.410', 'fear'], 'immoral': ['0.410', 'fear'], 'scar': ['0.406', 'fear'], 'rubble': ['0.406', 'fear'], 'weakly': ['0.406', 'fear'], 'distressed': ['0.406', 'fear'], 'nausea': ['0.406', 'fear'], 'whimper': ['0.406', 'fear'], 'stretcher': ['0.406', 'fear'], 'dismissal': ['0.406', 'fear'], 'ohdear': ['0.406', 'fear'], 'penance': ['0.406', 'fear'], 'omnipotence': ['0.406', 'fear'], 'insolvent': ['0.406', 'fear'], 'escape': ['0.406', 'fear'], 'resection': ['0.406', 'fear'], 'deport': ['0.406', 'fear'], 'disallowed': ['0.406', 'fear'], 'stressing': ['0.406', 'fear'], 'disgust': ['0.406', 'fear'], 'possession': ['0.406', 'fear'], 'ghetto': ['0.406', 'fear'], 'tearful': ['0.406', 'fear'], 'dominant': ['0.406', 'fear'], 'disorder': ['0.406', 'fear'], 'drones': ['0.406', 'fear'], 'wary': ['0.406', 'fear'], 'blackout': ['0.406', 'fear'], 'claw': ['0.406', 'fear'], 'disabled': ['0.406', 'fear'], 'screech': ['0.406', 'fear'], 'stressful': ['0.406', 'fear'], 'recklessness': ['0.406', 'fear'], 'squall': ['0.406', 'fear'], 'socialism': ['0.405', 'fear'], 'conspiracy': ['0.400', 'fear'], 'toughness': ['0.400', 'fear'], 'defection': ['0.398', 'fear'], 'absence': ['0.396', 'fear'], 'moan': ['0.394', 'fear'], 'crusade': ['0.392', 'fear'], 'discrimination': ['0.391', 'fear'], 'avoiding': ['0.391', 'fear'], 'concealment': ['0.391', 'fear'], 'overt': ['0.391', 'fear'], 'theocratic': ['0.391', 'fear'], 'unemployed': ['0.391', 'fear'], 'poverty': ['0.391', 'fear'], 'brimstone': ['0.391', 'fear'], 'cyst': ['0.391', 'fear'], 'radon': ['0.391', 'fear'], 'conquer': ['0.391', 'fear'], 'unrest': ['0.391', 'fear'], 'separation': ['0.391', 'fear'], 'suspicion': ['0.391', 'fear'], 'spaz': ['0.391', 'fear'], 'scarecrow': ['0.391', 'fear'], 'hardened': ['0.391', 'fear'], 'refutation': ['0.391', 'fear'], 'repellent': ['0.391', 'fear'], 'snare': ['0.391', 'fear'], 'cleave': ['0.391', 'fear'], 'gulp': ['0.391', 'fear'], 'nervy': ['0.391', 'fear'], 'submission': ['0.391', 'fear'], 'procedure': ['0.391', 'fear'], 'whatdoido': ['0.391', 'fear'], 'xanax': ['0.391', 'fear'], 'musket': ['0.391', 'fear'], 'descent': ['0.391', 'fear'], 'excitation': ['0.391', 'fear'], 'stifled': ['0.391', 'fear'], 'lose': ['0.391', 'fear'], 'diagnosis': ['0.391', 'fear'], 'urgency': ['0.391', 'fear'], 'mental': ['0.391', 'fear'], 'reject': ['0.391', 'fear'], 'exigent': ['0.391', 'fear'], 'insolvency': ['0.391', 'fear'], 'dubious': ['0.391', 'fear'], 'orc': ['0.388', 'fear'], 'outcast': ['0.388', 'fear'], 'throb': ['0.388', 'fear'], 'disapprove': ['0.380', 'fear'], 'dontpanic': ['0.378', 'fear'], 'withdrawals': ['0.377', 'fear'], 'plea': ['0.377', 'fear'], 'kerosene': ['0.375', 'fear'], 'bunker': ['0.375', 'fear'], 'escaped': ['0.375', 'fear'], 'dentistry': ['0.375', 'fear'], 'taunt': ['0.375', 'fear'], 'expose': ['0.375', 'fear'], 'bad': ['0.375', 'fear'], 'barricade': ['0.375', 'fear'], 'bankrupt': ['0.375', 'fear'], 'coldness': ['0.375', 'fear'], 'frigate': ['0.375', 'fear'], 'interrogate': ['0.375', 'fear'], 'grieve': ['0.375', 'fear'], 'ocd': ['0.375', 'fear'], 'dissolution': ['0.375', 'fear'], 'military': ['0.375', 'fear'], 'obi': ['0.375', 'fear'], 'sneaking': ['0.375', 'fear'], 'chimera': ['0.375', 'fear'], 'locust': ['0.375', 'fear'], 'embarrassment': ['0.375', 'fear'], 'mentalhealth': ['0.375', 'fear'], 'sultan': ['0.375', 'fear'], 'psychological': ['0.375', 'fear'], 'suspected': ['0.375', 'fear'], 'antsy': ['0.375', 'fear'], 'obligor': ['0.375', 'fear'], 'khan': ['0.375', 'fear'], 'nauseous': ['0.375', 'fear'], 'whirlpool': ['0.375', 'fear'], 'misconception': ['0.375', 'fear'], 'flu': ['0.375', 'fear'], 'chasm': ['0.375', 'fear'], 'edict': ['0.375', 'fear'], 'pressure': ['0.375', 'fear'], 'repellant': ['0.373', 'fear'], 'unknown': ['0.369', 'fear'], 'pare': ['0.367', 'fear'], 'jealousy': ['0.365', 'fear'], 'depreciation': ['0.359', 'fear'], 'contempt': ['0.359', 'fear'], 'government': ['0.359', 'fear'], 'desert': ['0.359', 'fear'], 'spike': ['0.359', 'fear'], 'onedge': ['0.359', 'fear'], 'formidable': ['0.359', 'fear'], 'exam': ['0.359', 'fear'], 'wasting': ['0.359', 'fear'], 'stint': ['0.359', 'fear'], 'sortie': ['0.359', 'fear'], 'bottomless': ['0.359', 'fear'], 'rejects': ['0.359', 'fear'], 'timid': ['0.359', 'fear'], 'burke': ['0.359', 'fear'], 'cur': ['0.359', 'fear'], 'jaundice': ['0.359', 'fear'], 'revolution': ['0.359', 'fear'], 'cautionary': ['0.359', 'fear'], 'dart': ['0.359', 'fear'], 'warned': ['0.359', 'fear'], 'pessimism': ['0.359', 'fear'], 'mug': ['0.359', 'fear'], 'difficult': ['0.359', 'fear'], 'measles': ['0.359', 'fear'], 'consternation': ['0.359', 'fear'], 'rebel': ['0.359', 'fear'], 'recurring': ['0.359', 'fear'], 'protestant': ['0.359', 'fear'], 'anomaly': ['0.359', 'fear'], 'headaches': ['0.359', 'fear'], 'mournful': ['0.359', 'fear'], 'mandamus': ['0.359', 'fear'], 'concerned': ['0.359', 'fear'], 'sectarian': ['0.359', 'fear'], 'conquest': ['0.359', 'fear'], 'bankruptcy': ['0.359', 'fear'], 'constrain': ['0.358', 'fear'], 'languishing': ['0.358', 'fear'], 'bane': ['0.356', 'fear'], 'warden': ['0.354', 'fear'], 'impeach': ['0.354', 'fear'], 'adverse': ['0.352', 'fear'], 'lawyer': ['0.349', 'fear'], 'libel': ['0.348', 'fear'], 'retrenchment': ['0.345', 'fear'], 'imminent': ['0.345', 'fear'], 'hiss': ['0.344', 'fear'], 'bearish': ['0.344', 'fear'], 'loneliness': ['0.344', 'fear'], 'obstacle': ['0.344', 'fear'], 'discipline': ['0.344', 'fear'], 'verdict': ['0.344', 'fear'], 'reckless': ['0.344', 'fear'], 'knell': ['0.344', 'fear'], 'swerve': ['0.344', 'fear'], 'taboo': ['0.344', 'fear'], 'parachute': ['0.344', 'fear'], 'sorrow': ['0.344', 'fear'], 'hesitation': ['0.344', 'fear'], 'servile': ['0.344', 'fear'], 'defy': ['0.344', 'fear'], 'launches': ['0.344', 'fear'], 'fanaticism': ['0.344', 'fear'], 'aaaaaaah': ['0.344', 'fear'], 'opium': ['0.344', 'fear'], 'shame': ['0.344', 'fear'], 'resistant': ['0.344', 'fear'], 'shaky': ['0.344', 'fear'], 'eel': ['0.344', 'fear'], 'opposed': ['0.344', 'fear'], 'mri': ['0.344', 'fear'], 'belittle': ['0.344', 'fear'], 'shortage': ['0.344', 'fear'], 'unjustifiable': ['0.344', 'fear'], 'recession': ['0.344', 'fear'], 'cutter': ['0.344', 'fear'], 'evade': ['0.344', 'fear'], 'pest': ['0.344', 'fear'], 'psych': ['0.344', 'fear'], 'avoidance': ['0.344', 'fear'], 'contentious': ['0.344', 'fear'], 'disrespectful': ['0.343', 'fear'], 'phalanx': ['0.342', 'fear'], 'creature': ['0.340', 'fear'], 'specter': ['0.331', 'fear'], 'mortgage': ['0.331', 'fear'], 'enigmatic': ['0.329', 'fear'], 'bugaboo': ['0.328', 'fear'], 'shutdown': ['0.328', 'fear'], 'spillin': ['0.328', 'fear'], 'encumbrance': ['0.328', 'fear'], 'caution': ['0.328', 'fear'], 'senseless': ['0.328', 'fear'], 'police': ['0.328', 'fear'], 'remove': ['0.328', 'fear'], 'dike': ['0.328', 'fear'], 'feeling': ['0.328', 'fear'], 'subordinate': ['0.328', 'fear'], 'quash': ['0.328', 'fear'], 'undesirable': ['0.328', 'fear'], 'smut': ['0.328', 'fear'], 'defendant': ['0.328', 'fear'], 'supremacy': ['0.328', 'fear'], 'loom': ['0.328', 'fear'], 'hives': ['0.328', 'fear'], 'mishap': ['0.328', 'fear'], 'uhoh': ['0.328', 'fear'], 'weirdo': ['0.328', 'fear'], 'wrongly': ['0.328', 'fear'], 'wimp': ['0.328', 'fear'], 'adrift': ['0.328', 'fear'], 'gladiator': ['0.328', 'fear'], 'outpost': ['0.328', 'fear'], 'ethereal': ['0.328', 'fear'], 'dominion': ['0.328', 'fear'], 'unlucky': ['0.328', 'fear'], 'shank': ['0.328', 'fear'], 'gametime': ['0.328', 'fear'], 'unsettled': ['0.327', 'fear'], 'scarce': ['0.327', 'fear'], 'antisocial': ['0.324', 'fear'], 'astray': ['0.320', 'fear'], 'vigilant': ['0.319', 'fear'], 'socialist': ['0.318', 'fear'], 'halter': ['0.318', 'fear'], 'pessimist': ['0.317', 'fear'], 'pacing': ['0.316', 'fear'], 'ordnance': ['0.312', 'fear'], 'whirlwind': ['0.312', 'fear'], 'seclusion': ['0.312', 'fear'], 'muzzle': ['0.312', 'fear'], 'trickery': ['0.312', 'fear'], 'collusion': ['0.312', 'fear'], 'nether': ['0.312', 'fear'], 'unkind': ['0.312', 'fear'], 'uneasy': ['0.312', 'fear'], 'valium': ['0.312', 'fear'], 'dentist': ['0.312', 'fear'], 'obsessing': ['0.312', 'fear'], 'powerful': ['0.312', 'fear'], 'cram': ['0.312', 'fear'], 'thorny': ['0.312', 'fear'], 'litigate': ['0.312', 'fear'], 'blight': ['0.312', 'fear'], 'therapist': ['0.312', 'fear'], 'deadline': ['0.312', 'fear'], 'opponent': ['0.312', 'fear'], 'ahhhhhhhh': ['0.312', 'fear'], 'wimpy': ['0.312', 'fear'], 'discontinuity': ['0.312', 'fear'], 'clinical': ['0.312', 'fear'], 'foreigner': ['0.312', 'fear'], 'chargeable': ['0.312', 'fear'], 'conspire': ['0.312', 'fear'], 'bewilderment': ['0.312', 'fear'], 'laxative': ['0.312', 'fear'], 'unexpected': ['0.312', 'fear'], 'overthinking': ['0.309', 'fear'], 'doubts': ['0.308', 'fear'], 'seriousness': ['0.305', 'fear'], 'irrational': ['0.305', 'fear'], 'erase': ['0.303', 'fear'], 'teasing': ['0.303', 'fear'], 'razor': ['0.303', 'fear'], 'sweating': ['0.298', 'fear'], 'medical': ['0.297', 'fear'], 'reluctant': ['0.297', 'fear'], 'adjudicate': ['0.297', 'fear'], 'timidity': ['0.297', 'fear'], 'depreciated': ['0.297', 'fear'], 'avoid': ['0.297', 'fear'], 'endless': ['0.297', 'fear'], 'unsurpassed': ['0.297', 'fear'], 'crouching': ['0.297', 'fear'], 'deflation': ['0.297', 'fear'], 'warning': ['0.297', 'fear'], 'flounder': ['0.297', 'fear'], 'giant': ['0.297', 'fear'], 'hide': ['0.297', 'fear'], 'bitch': ['0.297', 'fear'], 'swamp': ['0.297', 'fear'], 'auditor': ['0.297', 'fear'], 'dashed': ['0.297', 'fear'], 'inflation': ['0.297', 'fear'], 'bale': ['0.297', 'fear'], 'weighty': ['0.297', 'fear'], 'mislead': ['0.297', 'fear'], 'mage': ['0.297', 'fear'], 'deluge': ['0.287', 'fear'], 'disinformation': ['0.286', 'fear'], 'court': ['0.284', 'fear'], 'aversion': ['0.283', 'fear'], 'mistaken': ['0.281', 'fear'], 'cartridge': ['0.281', 'fear'], 'cautious': ['0.281', 'fear'], 'bigoted': ['0.281', 'fear'], 'unfriendly': ['0.281', 'fear'], 'wan': ['0.281', 'fear'], 'surveillance': ['0.281', 'fear'], 'hurryup': ['0.281', 'fear'], 'confusion': ['0.281', 'fear'], 'submitting': ['0.281', 'fear'], 'flying': ['0.281', 'fear'], 'challenge': ['0.281', 'fear'], 'oncoming': ['0.281', 'fear'], 'remains': ['0.281', 'fear'], 'notready': ['0.281', 'fear'], 'apache': ['0.281', 'fear'], 'rush': ['0.281', 'fear'], 'swelling': ['0.281', 'fear'], 'yelp': ['0.281', 'fear'], 'fortress': ['0.281', 'fear'], 'dwarfed': ['0.281', 'fear'], 'discourage': ['0.281', 'fear'], 'unruly': ['0.281', 'fear'], 'intense': ['0.279', 'fear'], 'noncompliance': ['0.276', 'fear'], 'alerts': ['0.276', 'fear'], 'planes': ['0.274', 'fear'], 'displeased': ['0.273', 'fear'], 'cop': ['0.273', 'fear'], 'unbridled': ['0.271', 'fear'], 'posse': ['0.266', 'fear'], 'blindfold': ['0.266', 'fear'], 'force': ['0.266', 'fear'], 'auditions': ['0.266', 'fear'], 'operation': ['0.266', 'fear'], 'hooded': ['0.266', 'fear'], 'banger': ['0.266', 'fear'], 'forgotten': ['0.266', 'fear'], 'crouch': ['0.266', 'fear'], 'flurries': ['0.266', 'fear'], 'shanghai': ['0.266', 'fear'], 'insomnia': ['0.266', 'fear'], 'blemish': ['0.266', 'fear'], 'problem': ['0.266', 'fear'], 'sneak': ['0.266', 'fear'], 'newjob': ['0.266', 'fear'], 'uncanny': ['0.266', 'fear'], 'hood': ['0.266', 'fear'], 'regiment': ['0.266', 'fear'], 'elevation': ['0.266', 'fear'], 'cloak': ['0.266', 'fear'], 'immerse': ['0.266', 'fear'], 'confession': ['0.266', 'fear'], 'acrobat': ['0.266', 'fear'], 'cupping': ['0.266', 'fear'], 'skid': ['0.266', 'fear'], 'crowds': ['0.266', 'fear'], 'belt': ['0.266', 'fear'], 'sentence': ['0.266', 'fear'], 'affront': ['0.266', 'fear'], 'spur': ['0.262', 'fear'], 'mortgagor': ['0.258', 'fear'], 'defense': ['0.258', 'fear'], 'hag': ['0.250', 'fear'], 'guard': ['0.250', 'fear'], 'run': ['0.250', 'fear'], 'cataract': ['0.250', 'fear'], 'unfamiliar': ['0.250', 'fear'], 'impatiently': ['0.250', 'fear'], 'standstill': ['0.250', 'fear'], 'rating': ['0.250', 'fear'], 'rascal': ['0.250', 'fear'], 'forewarned': ['0.250', 'fear'], 'breakingnews': ['0.250', 'fear'], 'doubt': ['0.250', 'fear'], 'obliging': ['0.250', 'fear'], 'foul': ['0.250', 'fear'], 'defend': ['0.250', 'fear'], 'spinster': ['0.250', 'fear'], 'repent': ['0.250', 'fear'], 'delusional': ['0.250', 'fear'], 'indecisive': ['0.250', 'fear'], 'insect': ['0.250', 'fear'], 'picket': ['0.250', 'fear'], 'indifference': ['0.250', 'fear'], 'sunk': ['0.250', 'fear'], 'jobinterview': ['0.250', 'fear'], 'react': ['0.250', 'fear'], 'swampy': ['0.250', 'fear'], 'scapegoat': ['0.250', 'fear'], 'stealthy': ['0.250', 'fear'], 'grounded': ['0.242', 'fear'], 'badhabit': ['0.242', 'fear'], 'highest': ['0.236', 'fear'], 'intercede': ['0.234', 'fear'], 'psychiatrist': ['0.234', 'fear'], 'buck': ['0.234', 'fear'], 'insomniac': ['0.234', 'fear'], 'dependence': ['0.234', 'fear'], 'scaffold': ['0.234', 'fear'], 'shrink': ['0.234', 'fear'], 'rekindle': ['0.234', 'fear'], 'overdrive': ['0.234', 'fear'], 'bee': ['0.234', 'fear'], 'sleepless': ['0.234', 'fear'], 'inequality': ['0.234', 'fear'], 'pinion': ['0.234', 'fear'], 'finalized': ['0.234', 'fear'], 'chicken': ['0.234', 'fear'], 'difficulty': ['0.234', 'fear'], 'instinctive': ['0.234', 'fear'], 'discontent': ['0.234', 'fear'], 'shelters': ['0.234', 'fear'], 'verge': ['0.234', 'fear'], 'aaaah': ['0.234', 'fear'], 'buzz': ['0.234', 'fear'], 'thatmoment': ['0.234', 'fear'], 'rushing': ['0.233', 'fear'], 'overslept': ['0.226', 'fear'], 'default': ['0.226', 'fear'], 'confessional': ['0.225', 'fear'], 'impatient': ['0.224', 'fear'], 'aga': ['0.219', 'fear'], 'unequal': ['0.219', 'fear'], 'jungle': ['0.219', 'fear'], 'thrill': ['0.219', 'fear'], 'bait': ['0.219', 'fear'], 'immigrant': ['0.219', 'fear'], 'waver': ['0.219', 'fear'], 'veer': ['0.219', 'fear'], 'lastminute': ['0.219', 'fear'], 'fingerscrossed': ['0.219', 'fear'], 'whatif': ['0.219', 'fear'], 'stingy': ['0.219', 'fear'], 'blues': ['0.219', 'fear'], 'tactics': ['0.219', 'fear'], 'revelations': ['0.219', 'fear'], 'sweat': ['0.219', 'fear'], 'cane': ['0.219', 'fear'], 'heft': ['0.219', 'fear'], 'overtired': ['0.218', 'fear'], 'prognosis': ['0.217', 'fear'], 'alertness': ['0.216', 'fear'], 'hyped': ['0.214', 'fear'], 'wishmeluck': ['0.212', 'fear'], 'somuchtodo': ['0.212', 'fear'], 'toomuchtodo': ['0.212', 'fear'], 'tryouts': ['0.210', 'fear'], 'clowns': ['0.207', 'fear'], 'fluctuation': ['0.204', 'fear'], 'examination': ['0.204', 'fear'], 'restless': ['0.204', 'fear'], 'intimacy': ['0.203', 'fear'], 'quail': ['0.203', 'fear'], 'overcome': ['0.203', 'fear'], 'acceptances': ['0.203', 'fear'], 'birth': ['0.203', 'fear'], 'phew': ['0.203', 'fear'], 'mighty': ['0.203', 'fear'], 'recesses': ['0.203', 'fear'], 'uphill': ['0.203', 'fear'], 'help': ['0.203', 'fear'], 'cross': ['0.203', 'fear'], 'hurry': ['0.203', 'fear'], 'fearless': ['0.200', 'fear'], 'change': ['0.198', 'fear'], 'ware': ['0.198', 'fear'], 'withstand': ['0.197', 'fear'], 'asap': ['0.191', 'fear'], 'overcoming': ['0.188', 'fear'], 'waitinggame': ['0.188', 'fear'], 'intrigue': ['0.188', 'fear'], 'stillness': ['0.188', 'fear'], 'owing': ['0.188', 'fear'], 'cautiously': ['0.188', 'fear'], 'watch': ['0.188', 'fear'], 'attorney': ['0.188', 'fear'], 'bug': ['0.188', 'fear'], 'fragile': ['0.188', 'fear'], 'unorganised': ['0.188', 'fear'], 'rule': ['0.188', 'fear'], 'advance': ['0.188', 'fear'], 'knots': ['0.188', 'fear'], 'tract': ['0.188', 'fear'], 'heighten': ['0.188', 'fear'], 'gnome': ['0.188', 'fear'], 'deadlines': ['0.188', 'fear'], 'elf': ['0.185', 'fear'], 'stillwaiting': ['0.184', 'fear'], 'lonely': ['0.183', 'fear'], 'slippery': ['0.181', 'fear'], 'interview': ['0.180', 'fear'], 'everyman': ['0.178', 'fear'], 'speculation': ['0.176', 'fear'], 'needtoknow': ['0.173', 'fear'], 'surprise': ['0.172', 'fear'], 'humanrights': ['0.172', 'fear'], 'shell': ['0.172', 'fear'], 'assessment': ['0.172', 'fear'], 'worship': ['0.172', 'fear'], 'chaff': ['0.172', 'fear'], 'composure': ['0.172', 'fear'], 'settlor': ['0.172', 'fear'], 'interviewer': ['0.172', 'fear'], 'unsure': ['0.172', 'fear'], 'competition': ['0.172', 'fear'], 'regulatory': ['0.172', 'fear'], 'readytogo': ['0.172', 'fear'], 'birch': ['0.172', 'fear'], 'rod': ['0.172', 'fear'], 'checkpoint': ['0.172', 'fear'], 'uncertain': ['0.172', 'fear'], 'less': ['0.167', 'fear'], 'coy': ['0.162', 'fear'], 'iris': ['0.160', 'fear'], 'intimately': ['0.156', 'fear'], 'sly': ['0.156', 'fear'], 'hearing': ['0.156', 'fear'], 'retirement': ['0.156', 'fear'], 'legalized': ['0.156', 'fear'], 'bases': ['0.156', 'fear'], 'ceasefire': ['0.156', 'fear'], 'audition': ['0.156', 'fear'], 'needit': ['0.156', 'fear'], 'medication': ['0.156', 'fear'], 'countdown': ['0.156', 'fear'], 'courageous': ['0.154', 'fear'], 'overthinker': ['0.153', 'fear'], 'backtrack': ['0.151', 'fear'], 'dawned': ['0.150', 'fear'], 'censor': ['0.149', 'fear'], 'bailiff': ['0.147', 'fear'], 'syllabus': ['0.147', 'fear'], 'gent': ['0.141', 'fear'], 'newcomer': ['0.141', 'fear'], 'rationality': ['0.141', 'fear'], 'imagination': ['0.141', 'fear'], 'cove': ['0.141', 'fear'], 'waiting': ['0.141', 'fear'], 'caricature': ['0.141', 'fear'], 'delay': ['0.141', 'fear'], 'eventuality': ['0.141', 'fear'], 'validity': ['0.141', 'fear'], 'shy': ['0.140', 'fear'], 'alreadyyyyy': ['0.140', 'fear'], 'symptom': ['0.133', 'fear'], 'excite': ['0.132', 'fear'], 'raccoon': ['0.127', 'fear'], 'campaigning': ['0.125', 'fear'], 'clown': ['0.125', 'fear'], 'marry': ['0.125', 'fear'], 'gameday': ['0.125', 'fear'], 'hawk': ['0.125', 'fear'], 'soready': ['0.125', 'fear'], 'sprite': ['0.125', 'fear'], 'tryout': ['0.125', 'fear'], 'prevent': ['0.125', 'fear'], 'needtorelax': ['0.125', 'fear'], 'swim': ['0.125', 'fear'], 'ahhh': ['0.125', 'fear'], 'pharmacy': ['0.123', 'fear'], 'sag': ['0.123', 'fear'], 'policeman': ['0.121', 'fear'], 'dreamt': ['0.111', 'fear'], 'lace': ['0.111', 'fear'], 'knot': ['0.109', 'fear'], 'incase': ['0.109', 'fear'], 'holiness': ['0.109', 'fear'], 'confidence': ['0.109', 'fear'], 'notoriety': ['0.109', 'fear'], 'homework': ['0.109', 'fear'], 'bigday': ['0.109', 'fear'], 'weight': ['0.109', 'fear'], 'pray': ['0.109', 'fear'], 'destination': ['0.109', 'fear'], 'slender': ['0.100', 'fear'], 'undecided': ['0.098', 'fear'], 'sympathetic': ['0.097', 'fear'], 'god': ['0.094', 'fear'], 'case': ['0.094', 'fear'], 'confident': ['0.094', 'fear'], 'helmet': ['0.094', 'fear'], 'poise': ['0.094', 'fear'], 'treat': ['0.094', 'fear'], 'locate': ['0.094', 'fear'], 'loyal': ['0.094', 'fear'], 'grades': ['0.094', 'fear'], 'counsellor': ['0.094', 'fear'], 'northeast': ['0.088', 'fear'], 'graduation': ['0.078', 'fear'], 'compassion': ['0.078', 'fear'], 'nurture': ['0.078', 'fear'], 'graded': ['0.075', 'fear'], 'mum': ['0.070', 'fear'], 'infant': ['0.067', 'fear'], 'youth': ['0.062', 'fear'], 'civilians': ['0.062', 'fear'], 'parade': ['0.062', 'fear'], 'gradschool': ['0.062', 'fear'], 'cash': ['0.062', 'fear'], 'civilian': ['0.062', 'fear'], 'cloudiness': ['0.062', 'fear'], 'journey': ['0.062', 'fear'], 'guidelines': ['0.062', 'fear'], 'soulmate': ['0.062', 'fear'], 'opera': ['0.057', 'fear'], 'synonymous': ['0.056', 'fear'], 'honest': ['0.047', 'fear'], 'praying': ['0.047', 'fear'], 'intelligence': ['0.038', 'fear'], 'volunteer': ['0.031', 'fear'], 'lines': ['0.031', 'fear'], 'romance': ['0.031', 'fear'], 'obey': ['0.016', 'fear']}
dic={'happiest': ['0.986', 'joy'], 'happiness': ['0.984', 'joy'], 'bliss': ['0.971', 'joy'], 'celebrating': ['0.970', 'joy'], 'jubilant': ['0.969', 'joy'], 'ecstatic': ['0.954', 'joy'], 'elation': ['0.944', 'joy'], 'beaming': ['0.938', 'joy'], 'bestdayever': ['0.938', 'joy'], 'loveee': ['0.932', 'joy'], 'celebration': ['0.929', 'joy'], 'awesomeness': ['0.926', 'joy'], 'joy': ['0.924', 'joy'], 'happily': ['0.922', 'joy'], 'fabulous': ['0.922', 'joy'], 'exuberance': ['0.922', 'joy'], 'excitement': ['0.922', 'joy'], 'joyous': ['0.922', 'joy'], 'makesmehappy': ['0.922', 'joy'], 'euphoria': ['0.922', 'joy'], 'lovee': ['0.920', 'joy'], 'gratitude': ['0.914', 'joy'], 'happydance': ['0.912', 'joy'], 'merriment': ['0.912', 'joy'], 'spectacular': ['0.912', 'joy'], 'overjoyed': ['0.909', 'joy'], 'purebliss': ['0.909', 'joy'], 'triumphant': ['0.907', 'joy'], 'lovelovelove': ['0.906', 'joy'], 'ecstasy': ['0.906', 'joy'], 'cheerful': ['0.906', 'joy'], 'cheer': ['0.897', 'joy'], 'elated': ['0.894', 'joy'], 'jolly': ['0.891', 'joy'], 'lovethis': ['0.891', 'joy'], 'peaceofmind': ['0.891', 'joy'], 'delighted': ['0.891', 'joy'], 'exhilaration': ['0.891', 'joy'], 'excitation': ['0.891', 'joy'], 'pleasures': ['0.891', 'joy'], 'laugh': ['0.891', 'joy'], 'marvelously': ['0.881', 'joy'], 'blissful': ['0.879', 'joy'], 'loving': ['0.879', 'joy'], 'outstanding': ['0.879', 'joy'], 'joyful': ['0.879', 'joy'], 'pleasurable': ['0.877', 'joy'], 'overthemoon': ['0.875', 'joy'], 'lovinglife': ['0.875', 'joy'], 'yaaaay': ['0.875', 'joy'], 'happyplace': ['0.875', 'joy'], 'iloveher': ['0.875', 'joy'], 'glee': ['0.875', 'joy'], 'enthusiastic': ['0.875', 'joy'], 'sohappy': ['0.868', 'joy'], 'superb': ['0.864', 'joy'], 'laughing': ['0.864', 'joy'], 'woohoo': ['0.864', 'joy'], 'wonderful': ['0.863', 'joy'], 'ilovechristmas': ['0.859', 'joy'], 'hooray': ['0.859', 'joy'], 'brilliant': ['0.859', 'joy'], 'cheering': ['0.859', 'joy'], 'glory': ['0.859', 'joy'], 'tearsofjoy': ['0.859', 'joy'], 'magnificent': ['0.859', 'joy'], 'hallelujah': ['0.859', 'joy'], 'yayyyy': ['0.859', 'joy'], 'celebrated': ['0.859', 'joy'], 'loved': ['0.859', 'joy'], 'exciting': ['0.853', 'joy'], 'heavenly': ['0.853', 'joy'], 'thrilled': ['0.851', 'joy'], 'mademyday': ['0.848', 'joy'], 'hohoho': ['0.845', 'joy'], 'wonderfully': ['0.844', 'joy'], 'blessing': ['0.844', 'joy'], 'favoriteholiday': ['0.844', 'joy'], 'celebrate': ['0.844', 'joy'], 'celebrations': ['0.833', 'joy'], 'blessed': ['0.833', 'joy'], 'festive': ['0.833', 'joy'], 'sweetness': ['0.833', 'joy'], 'paradise': ['0.833', 'joy'], 'marvellous': ['0.833', 'joy'], 'compliment': ['0.831', 'joy'], 'enchanting': ['0.828', 'joy'], 'smiling': ['0.828', 'joy'], 'allsmiles': ['0.828', 'joy'], 'love': ['0.828', 'joy'], 'homesweethome': ['0.826', 'joy'], 'thankyougod': ['0.824', 'joy'], 'marvelous': ['0.824', 'joy'], 'laughter': ['0.824', 'joy'], 'goodmood': ['0.819', 'joy'], 'happyheart': ['0.818', 'joy'], 'joys': ['0.818', 'joy'], 'sensational': ['0.818', 'joy'], 'celebratory': ['0.818', 'joy'], 'excellence': ['0.818', 'joy'], 'delightful': ['0.818', 'joy'], 'goodness': ['0.818', 'joy'], 'excited': ['0.818', 'joy'], 'rejoicing': ['0.818', 'joy'], 'greatful': ['0.816', 'joy'], 'jovial': ['0.814', 'joy'], 'glorious': ['0.812', 'joy'], 'victorious': ['0.812', 'joy'], 'excellent': ['0.812', 'joy'], 'bonanza': ['0.812', 'joy'], 'rejoice': ['0.812', 'joy'], 'splendid': ['0.812', 'joy'], 'enjoy': ['0.812', 'joy'], 'lovemaking': ['0.812', 'joy'], 'greatday': ['0.812', 'joy'], 'smiley': ['0.812', 'joy'], 'goodtimes': ['0.811', 'joy'], 'whatmakesmesmile': ['0.811', 'joy'], 'happyday': ['0.809', 'joy'], 'myfavorite': ['0.804', 'joy'], 'yeahhhh': ['0.803', 'joy'], 'gladness': ['0.803', 'joy'], 'yayyy': ['0.803', 'joy'], 'pleasure': ['0.803', 'joy'], 'thankyoulord': ['0.803', 'joy'], 'giggle': ['0.802', 'joy'], 'lovinlife': ['0.797', 'joy'], 'yesss': ['0.797', 'joy'], 'happytweet': ['0.797', 'joy'], 'success': ['0.797', 'joy'], 'dancing': ['0.797', 'joy'], 'lovemylife': ['0.797', 'joy'], 'happier': ['0.797', 'joy'], 'magnificence': ['0.797', 'joy'], 'grateful': ['0.789', 'joy'], 'happy': ['0.788', 'joy'], 'amuse': ['0.788', 'joy'], 'splendor': ['0.788', 'joy'], 'fun': ['0.788', 'joy'], 'glorify': ['0.781', 'joy'], 'solucky': ['0.781', 'joy'], 'glad': ['0.781', 'joy'], 'enchanted': ['0.781', 'joy'], 'sothankful': ['0.781', 'joy'], 'radiant': ['0.781', 'joy'], 'beautiful': ['0.781', 'joy'], 'giggling': ['0.781', 'joy'], 'perfection': ['0.779', 'joy'], 'christmassy': ['0.779', 'joy'], 'heavens': ['0.779', 'joy'], 'romance': ['0.779', 'joy'], 'thrilling': ['0.776', 'joy'], 'happyvalentinesday': ['0.773', 'joy'], 'entertain': ['0.773', 'joy'], 'cheered': ['0.773', 'joy'], 'positivity': ['0.773', 'joy'], 'congrats': ['0.773', 'joy'], 'cheers': ['0.773', 'joy'], 'lovable': ['0.773', 'joy'], 'miraculous': ['0.773', 'joy'], 'fiesta': ['0.773', 'joy'], 'funday': ['0.772', 'joy'], 'enjoying': ['0.771', 'joy'], 'amused': ['0.766', 'joy'], 'smiles': ['0.766', 'joy'], 'lifeisgood': ['0.766', 'joy'], 'thebest': ['0.766', 'joy'], 'cuddling': ['0.766', 'joy'], 'sosweet': ['0.766', 'joy'], 'christmasspirit': ['0.766', 'joy'], 'goodfeeling': ['0.766', 'joy'], 'delight': ['0.765', 'joy'], 'orgasm': ['0.765', 'joy'], 'party': ['0.765', 'joy'], 'positive': ['0.761', 'joy'], 'enlighten': ['0.758', 'joy'], 'cheerfulness': ['0.758', 'joy'], 'miracles': ['0.758', 'joy'], 'sweetheart': ['0.758', 'joy'], 'giddy': ['0.757', 'joy'], 'christmastime': ['0.757', 'joy'], 'pleasing': ['0.750', 'joy'], 'gratify': ['0.750', 'joy'], 'smile': ['0.750', 'joy'], 'laughs': ['0.750', 'joy'], 'greatness': ['0.750', 'joy'], 'friendliness': ['0.750', 'joy'], 'happyholidays': ['0.750', 'joy'], 'romantic': ['0.750', 'joy'], 'blessings': ['0.750', 'joy'], 'tistheseason': ['0.750', 'joy'], 'frolic': ['0.748', 'joy'], 'positiveenergy': ['0.742', 'joy'], 'rewarding': ['0.742', 'joy'], 'magical': ['0.742', 'joy'], 'miracle': ['0.742', 'joy'], 'selflove': ['0.742', 'joy'], 'jubilee': ['0.742', 'joy'], 'triumph': ['0.742', 'joy'], 'goodvibes': ['0.742', 'joy'], 'enthusiasm': ['0.742', 'joy'], 'feelgood': ['0.736', 'joy'], 'prosperity': ['0.735', 'joy'], 'passionate': ['0.734', 'joy'], 'admiration': ['0.734', 'joy'], 'feelinggood': ['0.734', 'joy'], 'tgif': ['0.734', 'joy'], 'victory': ['0.734', 'joy'], 'enchant': ['0.734', 'joy'], 'vivacious': ['0.734', 'joy'], 'luxurious': ['0.734', 'joy'], 'behappy': ['0.734', 'joy'], 'greatnight': ['0.734', 'joy'], 'goodday': ['0.734', 'joy'], 'glorification': ['0.733', 'joy'], 'glowing': ['0.729', 'joy'], 'sing': ['0.729', 'joy'], 'breathtaking': ['0.728', 'joy'], 'yessss': ['0.728', 'joy'], 'fulfillment': ['0.728', 'joy'], 'atpeace': ['0.727', 'joy'], 'hurrah': ['0.727', 'joy'], 'merry': ['0.727', 'joy'], 'santa': ['0.727', 'joy'], 'award': ['0.727', 'joy'], 'christmasbreak': ['0.727', 'joy'], 'thankful': ['0.727', 'joy'], 'cheery': ['0.727', 'joy'], 'win': ['0.727', 'joy'], 'pleased': ['0.725', 'joy'], 'inspiration': ['0.725', 'joy'], 'radiance': ['0.725', 'joy'], 'uplift': ['0.723', 'joy'], 'optimistic': ['0.723', 'joy'], 'holidays': ['0.721', 'joy'], 'thrill': ['0.721', 'joy'], 'heaven': ['0.721', 'joy'], 'godisgreat': ['0.721', 'joy'], 'lucky': ['0.721', 'joy'], 'amusement': ['0.719', 'joy'], 'congratulatory': ['0.719', 'joy'], 'harmony': ['0.719', 'joy'], 'brighten': ['0.719', 'joy'], 'lover': ['0.719', 'joy'], 'perfect': ['0.719', 'joy'], 'lovely': ['0.719', 'joy'], 'thriving': ['0.719', 'joy'], 'praising': ['0.719', 'joy'], 'utopian': ['0.719', 'joy'], 'xmas': ['0.719', 'joy'], 'heartfelt': ['0.719', 'joy'], 'luxury': ['0.712', 'joy'], 'treasures': ['0.712', 'joy'], 'magic': ['0.712', 'joy'], 'bestfeeling': ['0.712', 'joy'], 'merrychristmas': ['0.712', 'joy'], 'achievement': ['0.712', 'joy'], 'holiday': ['0.712', 'joy'], 'yay': ['0.712', 'joy'], 'luckiest': ['0.712', 'joy'], 'intimate': ['0.710', 'joy'], 'yaaay': ['0.706', 'joy'], 'chuckle': ['0.706', 'joy'], 'rave': ['0.706', 'joy'], 'soblessed': ['0.706', 'joy'], 'proud': ['0.704', 'joy'], 'cherish': ['0.703', 'joy'], 'sweetest': ['0.703', 'joy'], 'amazingly': ['0.703', 'joy'], 'optimism': ['0.703', 'joy'], 'fuckyeah': ['0.703', 'joy'], 'goodnews': ['0.703', 'joy'], 'cuddled': ['0.703', 'joy'], 'satisfying': ['0.703', 'joy'], 'beautification': ['0.703', 'joy'], 'truelove': ['0.703', 'joy'], 'lovelife': ['0.703', 'joy'], 'gooood': ['0.703', 'joy'], 'goodlife': ['0.703', 'joy'], 'appreciates': ['0.703', 'joy'], 'winning': ['0.703', 'joy'], 'yaay': ['0.700', 'joy'], 'entertained': ['0.700', 'joy'], 'excite': ['0.697', 'joy'], 'newbeginnings': ['0.693', 'joy'], 'praisejesus': ['0.691', 'joy'], 'birthday': ['0.691', 'joy'], 'exquisite': ['0.688', 'joy'], 'content': ['0.688', 'joy'], 'godsend': ['0.688', 'joy'], 'thankyoujesus': ['0.688', 'joy'], 'adoration': ['0.688', 'joy'], 'angelic': ['0.688', 'joy'], 'greatfriends': ['0.688', 'joy'], 'favorite': ['0.688', 'joy'], 'metime': ['0.688', 'joy'], 'honored': ['0.688', 'joy'], 'holidayseason': ['0.688', 'joy'], 'entertaining': ['0.688', 'joy'], 'majestic': ['0.682', 'joy'], 'brightens': ['0.682', 'joy'], 'exaltation': ['0.682', 'joy'], 'goodhealth': ['0.682', 'joy'], 'smiled': ['0.682', 'joy'], 'bestfriends': ['0.682', 'joy'], 'memoriesiwontforget': ['0.682', 'joy'], 'precious': ['0.682', 'joy'], 'luscious': ['0.682', 'joy'], 'appreciated': ['0.682', 'joy'], 'tranquility': ['0.679', 'joy'], 'embrace': ['0.676', 'joy'], 'marry': ['0.676', 'joy'], 'positively': ['0.676', 'joy'], 'grin': ['0.672', 'joy'], 'giggles': ['0.672', 'joy'], 'enliven': ['0.672', 'joy'], 'bday': ['0.672', 'joy'], 'relaxation': ['0.672', 'joy'], 'hug': ['0.672', 'joy'], 'hilarious': ['0.672', 'joy'], 'contentment': ['0.672', 'joy'], 'weeeee': ['0.672', 'joy'], 'dearest': ['0.672', 'joy'], 'accomplished': ['0.672', 'joy'], 'fulfilled': ['0.667', 'joy'], 'adore': ['0.667', 'joy'], 'bountiful': ['0.667', 'joy'], 'victor': ['0.667', 'joy'], 'boisterous': ['0.667', 'joy'], 'fulfill': ['0.664', 'joy'], 'cuddles': ['0.662', 'joy'], 'prosperous': ['0.660', 'joy'], 'serenity': ['0.656', 'joy'], 'glow': ['0.656', 'joy'], 'encouraged': ['0.656', 'joy'], 'christmaseve': ['0.656', 'joy'], 'appreciation': ['0.656', 'joy'], 'happynewyear': ['0.656', 'joy'], 'satisfy': ['0.656', 'joy'], 'innerpeace': ['0.656', 'joy'], 'captivate': ['0.656', 'joy'], 'besties': ['0.656', 'joy'], 'romanticism': ['0.656', 'joy'], 'humor': ['0.656', 'joy'], 'pleasant': ['0.656', 'joy'], 'satisfaction': ['0.652', 'joy'], 'praised': ['0.652', 'joy'], 'abundance': ['0.652', 'joy'], 'treasure': ['0.652', 'joy'], 'praises': ['0.652', 'joy'], 'engaged': ['0.652', 'joy'], 'relaxing': ['0.652', 'joy'], 'fortunes': ['0.652', 'joy'], 'nothingbetter': ['0.647', 'joy'], 'complement': ['0.647', 'joy'], 'affection': ['0.647', 'joy'], 'relieved': ['0.647', 'joy'], 'carnival': ['0.643', 'joy'], 'uplifting': ['0.641', 'joy'], 'divine': ['0.641', 'joy'], 'champion': ['0.641', 'joy'], 'thanksgiving': ['0.641', 'joy'], 'achieve': ['0.641', 'joy'], 'jackpot': ['0.641', 'joy'], 'priceless': ['0.641', 'joy'], 'saintly': ['0.641', 'joy'], 'sensuality': ['0.641', 'joy'], 'wedding': ['0.641', 'joy'], 'harmoniously': ['0.641', 'joy'], 'honeymoon': ['0.636', 'joy'], 'exalt': ['0.636', 'joy'], 'twinkle': ['0.636', 'joy'], 'cuddle': ['0.636', 'joy'], 'felicity': ['0.636', 'joy'], 'peaceful': ['0.636', 'joy'], 'yayy': ['0.636', 'joy'], 'winner': ['0.636', 'joy'], 'reverie': ['0.636', 'joy'], 'climax': ['0.636', 'joy'], 'comforting': ['0.636', 'joy'], 'xoxo': ['0.634', 'joy'], 'reward': ['0.625', 'joy'], 'gorgeous': ['0.625', 'joy'], 'praisegod': ['0.625', 'joy'], 'generosity': ['0.625', 'joy'], 'hearts': ['0.625', 'joy'], 'stargazing': ['0.625', 'joy'], 'snuggling': ['0.625', 'joy'], 'fondness': ['0.625', 'joy'], 'amusing': ['0.625', 'joy'], 'sweet': ['0.625', 'joy'], 'brighter': ['0.625', 'joy'], 'festival': ['0.625', 'joy'], 'sex': ['0.622', 'joy'], 'kind': ['0.621', 'joy'], 'parade': ['0.621', 'joy'], 'genial': ['0.621', 'joy'], 'applause': ['0.621', 'joy'], 'beauty': ['0.621', 'joy'], 'fulfilling': ['0.618', 'joy'], 'aspiring': ['0.618', 'joy'], 'newlife': ['0.618', 'joy'], 'godbless': ['0.616', 'joy'], 'virtuous': ['0.613', 'joy'], 'kiss': ['0.610', 'joy'], 'rainbows': ['0.609', 'joy'], 'generous': ['0.609', 'joy'], 'christmas': ['0.609', 'joy'], 'enlightenment': ['0.609', 'joy'], 'winnings': ['0.609', 'joy'], 'playful': ['0.609', 'joy'], 'super': ['0.609', 'joy'], 'awards': ['0.609', 'joy'], 'praiseworthy': ['0.609', 'joy'], 'rekindle': ['0.609', 'joy'], 'adorable': ['0.609', 'joy'], 'elegance': ['0.609', 'joy'], 'independence': ['0.607', 'joy'], 'amour': ['0.607', 'joy'], 'kindness': ['0.606', 'joy'], 'inspired': ['0.606', 'joy'], 'wonder': ['0.606', 'joy'], 'successful': ['0.606', 'joy'], 'heheh': ['0.606', 'joy'], 'onelove': ['0.606', 'joy'], 'hilarity': ['0.606', 'joy'], 'freely': ['0.606', 'joy'], 'surprises': ['0.606', 'joy'], 'entertainment': ['0.603', 'joy'], 'passion': ['0.603', 'joy'], 'whimsical': ['0.603', 'joy'], 'beautify': ['0.601', 'joy'], 'stressfree': ['0.601', 'joy'], 'sunrise': ['0.600', 'joy'], 'godisgood': ['0.597', 'joy'], 'revere': ['0.594', 'joy'], 'snuggled': ['0.594', 'joy'], 'accomplishment': ['0.594', 'joy'], 'jesus': ['0.594', 'joy'], 'angel': ['0.594', 'joy'], 'goodmusic': ['0.594', 'joy'], 'inspire': ['0.594', 'joy'], 'flirt': ['0.594', 'joy'], 'thankgod': ['0.594', 'joy'], 'whoo': ['0.594', 'joy'], 'goodies': ['0.594', 'joy'], 'peacefully': ['0.594', 'joy'], 'fanfare': ['0.594', 'joy'], 'friendship': ['0.591', 'joy'], 'heroic': ['0.591', 'joy'], 'summer': ['0.591', 'joy'], 'fortune': ['0.591', 'joy'], 'highest': ['0.591', 'joy'], 'singing': ['0.591', 'joy'], 'exalted': ['0.591', 'joy'], 'woot': ['0.588', 'joy'], 'contented': ['0.588', 'joy'], 'overflowing': ['0.588', 'joy'], 'rollicking': ['0.588', 'joy'], 'hope': ['0.586', 'joy'], 'accolade': ['0.586', 'joy'], 'cozy': ['0.583', 'joy'], 'delicious': ['0.579', 'joy'], 'ambition': ['0.578', 'joy'], 'friendly': ['0.578', 'joy'], 'praise': ['0.578', 'joy'], 'raving': ['0.578', 'joy'], 'sensuous': ['0.578', 'joy'], 'picturesque': ['0.578', 'joy'], 'thelife': ['0.578', 'joy'], 'everlasting': ['0.578', 'joy'], 'darling': ['0.578', 'joy'], 'sparkle': ['0.577', 'joy'], 'yeahhh': ['0.576', 'joy'], 'flattering': ['0.576', 'joy'], 'succeeding': ['0.576', 'joy'], 'peace': ['0.576', 'joy'], 'heroism': ['0.576', 'joy'], 'luckygirl': ['0.576', 'joy'], 'sensual': ['0.576', 'joy'], 'grace': ['0.576', 'joy'], 'special': ['0.574', 'joy'], 'livelife': ['0.574', 'joy'], 'tantalizing': ['0.572', 'joy'], 'pumped': ['0.567', 'joy'], 'relax': ['0.562', 'joy'], 'hero': ['0.562', 'joy'], 'sweets': ['0.562', 'joy'], 'admirable': ['0.562', 'joy'], 'yey': ['0.562', 'joy'], 'surprise': ['0.562', 'joy'], 'hugs': ['0.562', 'joy'], 'prosper': ['0.562', 'joy'], 'revels': ['0.562', 'joy'], 'sunny': ['0.562', 'joy'], 'prevail': ['0.562', 'joy'], 'liking': ['0.562', 'joy'], 'humorous': ['0.562', 'joy'], 'worthwhile': ['0.562', 'joy'], 'superstar': ['0.562', 'joy'], 'bless': ['0.561', 'joy'], 'favorable': ['0.561', 'joy'], 'tenderness': ['0.561', 'joy'], 'newyear': ['0.561', 'joy'], 'freedom': ['0.561', 'joy'], 'masterpiece': ['0.561', 'joy'], 'dreams': ['0.559', 'joy'], 'hopeful': ['0.559', 'joy'], 'home': ['0.559', 'joy'], 'cruising': ['0.556', 'joy'], 'gracias': ['0.554', 'joy'], 'faithfulness': ['0.553', 'joy'], 'eagerness': ['0.552', 'joy'], 'closeness': ['0.552', 'joy'], 'sunshine': ['0.551', 'joy'], 'comfy': ['0.551', 'joy'], 'gift': ['0.547', 'joy'], 'bonus': ['0.547', 'joy'], 'daughter': ['0.547', 'joy'], 'vacation': ['0.547', 'joy'], 'confidence': ['0.547', 'joy'], 'zeal': ['0.547', 'joy'], 'astonishment': ['0.547', 'joy'], 'heart': ['0.547', 'joy'], 'completion': ['0.547', 'joy'], 'gifts': ['0.547', 'joy'], 'mistletoe': ['0.547', 'joy'], 'elite': ['0.547', 'joy'], 'good': ['0.547', 'joy'], 'celestial': ['0.547', 'joy'], 'illuminate': ['0.547', 'joy'], 'lifted': ['0.547', 'joy'], 'goodmorning': ['0.547', 'joy'], 'charmed': ['0.546', 'joy'], 'encouragement': ['0.545', 'joy'], 'sublime': ['0.545', 'joy'], 'dance': ['0.545', 'joy'], 'recreation': ['0.545', 'joy'], 'gush': ['0.545', 'joy'], 'god': ['0.545', 'joy'], 'free': ['0.544', 'joy'], 'freshstart': ['0.544', 'joy'], 'savior': ['0.543', 'joy'], 'sanctuary': ['0.541', 'joy'], 'grandchildren': ['0.540', 'joy'], 'wellness': ['0.537', 'joy'], 'revel': ['0.534', 'joy'], 'alive': ['0.531', 'joy'], 'bridal': ['0.531', 'joy'], 'inspirational': ['0.531', 'joy'], 'vitality': ['0.531', 'joy'], 'liberation': ['0.531', 'joy'], 'holiness': ['0.531', 'joy'], 'firstborn': ['0.531', 'joy'], 'money': ['0.531', 'joy'], 'rainbow': ['0.531', 'joy'], 'dayoff': ['0.531', 'joy'], 'serene': ['0.531', 'joy'], 'confident': ['0.531', 'joy'], 'soothing': ['0.531', 'joy'], 'music': ['0.531', 'joy'], 'matrimony': ['0.531', 'joy'], 'soar': ['0.531', 'joy'], 'savor': ['0.531', 'joy'], 'fab': ['0.531', 'joy'], 'mastery': ['0.530', 'joy'], 'warmth': ['0.530', 'joy'], 'elegant': ['0.530', 'joy'], 'glimmer': ['0.530', 'joy'], 'blossom': ['0.530', 'joy'], 'illumination': ['0.530', 'joy'], 'welcomed': ['0.530', 'joy'], 'treat': ['0.529', 'joy'], 'faithful': ['0.529', 'joy'], 'snuggles': ['0.522', 'joy'], 'laurels': ['0.521', 'joy'], 'commendable': ['0.519', 'joy'], 'strengthening': ['0.516', 'joy'], 'succeed': ['0.516', 'joy'], 'aspire': ['0.516', 'joy'], 'abundant': ['0.516', 'joy'], 'powerful': ['0.516', 'joy'], 'almighty': ['0.516', 'joy'], 'jingle': ['0.516', 'joy'], 'silly': ['0.516', 'joy'], 'remarkable': ['0.516', 'joy'], 'zest': ['0.516', 'joy'], 'pride': ['0.516', 'joy'], 'brotherly': ['0.516', 'joy'], 'greeted': ['0.516', 'joy'], 'presents': ['0.516', 'joy'], 'resplendent': ['0.516', 'joy'], 'fancy': ['0.516', 'joy'], 'noschool': ['0.516', 'joy'], 'leisure': ['0.515', 'joy'], 'vivid': ['0.515', 'joy'], 'thanking': ['0.515', 'joy'], 'therapeutic': ['0.515', 'joy'], 'familytime': ['0.515', 'joy'], 'zen': ['0.515', 'joy'], 'reunited': ['0.515', 'joy'], 'animated': ['0.515', 'joy'], 'comfort': ['0.515', 'joy'], 'princely': ['0.515', 'joy'], 'shining': ['0.515', 'joy'], 'complete': ['0.515', 'joy'], 'kudos': ['0.515', 'joy'], 'payday': ['0.515', 'joy'], 'cutie': ['0.515', 'joy'], 'coronation': ['0.515', 'joy'], 'spirit': ['0.515', 'joy'], 'newme': ['0.515', 'joy'], 'kid': ['0.515', 'joy'], 'marriage': ['0.514', 'joy'], 'relationship': ['0.514', 'joy'], 'daymade': ['0.514', 'joy'], 'godly': ['0.514', 'joy'], 'spouse': ['0.507', 'joy'], 'intimately': ['0.507', 'joy'], 'achieved': ['0.500', 'joy'], 'soulful': ['0.500', 'joy'], 'welcoming': ['0.500', 'joy'], 'satisfied': ['0.500', 'joy'], 'family': ['0.500', 'joy'], 'meritorious': ['0.500', 'joy'], 'purr': ['0.500', 'joy'], 'motherhood': ['0.500', 'joy'], 'carefree': ['0.500', 'joy'], 'gem': ['0.500', 'joy'], 'excel': ['0.500', 'joy'], 'healthy': ['0.500', 'joy'], 'surreal': ['0.500', 'joy'], 'diamond': ['0.500', 'joy'], 'charitable': ['0.500', 'joy'], 'inviting': ['0.500', 'joy'], 'erotic': ['0.500', 'joy'], 'memorable': ['0.500', 'joy'], 'veracity': ['0.500', 'joy'], 'friends': ['0.500', 'joy'], 'stressrelief': ['0.500', 'joy'], 'holyspirit': ['0.500', 'joy'], 'respect': ['0.500', 'joy'], 'beach': ['0.500', 'joy'], 'nature': ['0.500', 'joy'], 'dignity': ['0.500', 'joy'], 'bloom': ['0.500', 'joy'], 'accomplish': ['0.500', 'joy'], 'christ': ['0.500', 'joy'], 'encourage': ['0.500', 'joy'], 'teamjesus': ['0.500', 'joy'], 'visionary': ['0.500', 'joy'], 'baby': ['0.500', 'joy'], 'refreshed': ['0.500', 'joy'], 'aura': ['0.493', 'joy'], 'health': ['0.493', 'joy'], 'liberty': ['0.486', 'joy'], 'oasis': ['0.486', 'joy'], 'yehey': ['0.486', 'joy'], 'approved': ['0.486', 'joy'], 'rapture': ['0.485', 'joy'], 'loyal': ['0.485', 'joy'], 'aspiration': ['0.485', 'joy'], 'inseparable': ['0.485', 'joy'], 'betrothed': ['0.485', 'joy'], 'privileged': ['0.485', 'joy'], 'crescendo': ['0.485', 'joy'], 'crowning': ['0.485', 'joy'], 'gentle': ['0.485', 'joy'], 'liberate': ['0.485', 'joy'], 'nocomplaints': ['0.485', 'joy'], 'engaging': ['0.485', 'joy'], 'bounty': ['0.485', 'joy'], 'prestige': ['0.485', 'joy'], 'yummy': ['0.484', 'joy'], 'chocolate': ['0.484', 'joy'], 'desire': ['0.484', 'joy'], 'heyday': ['0.484', 'joy'], 'selfworth': ['0.484', 'joy'], 'dream': ['0.484', 'joy'], 'transcendence': ['0.484', 'joy'], 'luck': ['0.484', 'joy'], 'creativity': ['0.484', 'joy'], 'bouquet': ['0.484', 'joy'], 'aloha': ['0.484', 'joy'], 'trophy': ['0.484', 'joy'], 'fete': ['0.484', 'joy'], 'destiny': ['0.484', 'joy'], 'datenight': ['0.484', 'joy'], 'boyfriend': ['0.480', 'joy'], 'commemoration': ['0.479', 'joy'], 'intelligence': ['0.477', 'joy'], 'readiness': ['0.473', 'joy'], 'friend': ['0.471', 'joy'], 'enthusiast': ['0.471', 'joy'], 'bride': ['0.471', 'joy'], 'lush': ['0.470', 'joy'], 'inheritance': ['0.470', 'joy'], 'calming': ['0.470', 'joy'], 'soothe': ['0.470', 'joy'], 'adventure': ['0.470', 'joy'], 'kiddo': ['0.470', 'joy'], 'nostalgia': ['0.470', 'joy'], 'tickle': ['0.470', 'joy'], 'massage': ['0.470', 'joy'], 'purring': ['0.469', 'joy'], 'bonding': ['0.469', 'joy'], 'eternal': ['0.469', 'joy'], 'benevolence': ['0.469', 'joy'], 'nurture': ['0.469', 'joy'], 'giving': ['0.469', 'joy'], 'princess': ['0.469', 'joy'], 'pretty': ['0.469', 'joy'], 'amicable': ['0.469', 'joy'], 'getaway': ['0.469', 'joy'], 'goals': ['0.469', 'joy'], 'humanitarian': ['0.469', 'joy'], 'luster': ['0.469', 'joy'], 'bridegroom': ['0.469', 'joy'], 'pray': ['0.469', 'joy'], 'rest': ['0.469', 'joy'], 'heartily': ['0.469', 'joy'], 'child': ['0.466', 'joy'], 'salutary': ['0.465', 'joy'], 'invite': ['0.457', 'joy'], 'beam': ['0.456', 'joy'], 'reminiscing': ['0.456', 'joy'], 'tropical': ['0.455', 'joy'], 'befriend': ['0.455', 'joy'], 'hee': ['0.455', 'joy'], 'ceremony': ['0.455', 'joy'], 'friday': ['0.455', 'joy'], 'elevation': ['0.455', 'joy'], 'sonice': ['0.455', 'joy'], 'aesthetics': ['0.455', 'joy'], 'scholarship': ['0.455', 'joy'], 'kindred': ['0.455', 'joy'], 'mindfulness': ['0.455', 'joy'], 'freshair': ['0.455', 'joy'], 'birth': ['0.453', 'joy'], 'scenery': ['0.453', 'joy'], 'faith': ['0.453', 'joy'], 'namaste': ['0.453', 'joy'], 'vindication': ['0.453', 'joy'], 'allure': ['0.453', 'joy'], 'noworries': ['0.453', 'joy'], 'commemorate': ['0.453', 'joy'], 'approve': ['0.453', 'joy'], 'forgiveness': ['0.453', 'joy'], 'waterfall': ['0.453', 'joy'], 'journey': ['0.447', 'joy'], 'meditation': ['0.446', 'joy'], 'relaxed': ['0.442', 'joy'], 'weekend': ['0.441', 'joy'], 'tranquil': ['0.441', 'joy'], 'tender': ['0.441', 'joy'], 'present': ['0.441', 'joy'], 'righteousness': ['0.439', 'joy'], 'sharing': ['0.439', 'joy'], 'lyrical': ['0.439', 'joy'], 'esteem': ['0.439', 'joy'], 'nostalgic': ['0.439', 'joy'], 'prayer': ['0.439', 'joy'], 'unbeaten': ['0.438', 'joy'], 'share': ['0.438', 'joy'], 'eager': ['0.438', 'joy'], 'strength': ['0.438', 'joy'], 'meditate': ['0.438', 'joy'], 'newday': ['0.438', 'joy'], 'husband': ['0.438', 'joy'], 'life': ['0.438', 'joy'], 'sonnet': ['0.438', 'joy'], 'relief': ['0.438', 'joy'], 'mighty': ['0.438', 'joy'], 'warm': ['0.429', 'joy'], 'winterbreak': ['0.429', 'joy'], 'movingforward': ['0.429', 'joy'], 'buddy': ['0.427', 'joy'], 'oneness': ['0.426', 'joy'], 'medal': ['0.424', 'joy'], 'unsurpassed': ['0.424', 'joy'], 'carols': ['0.424', 'joy'], 'candlelight': ['0.424', 'joy'], 'amen': ['0.424', 'joy'], 'reverence': ['0.424', 'joy'], 'ejaculation': ['0.424', 'joy'], 'healthful': ['0.424', 'joy'], 'unconstrained': ['0.424', 'joy'], 'thelittlethings': ['0.424', 'joy'], 'wealth': ['0.422', 'joy'], 'graduation': ['0.422', 'joy'], 'glitter': ['0.422', 'joy'], 'lml': ['0.422', 'joy'], 'ease': ['0.422', 'joy'], 'sledding': ['0.422', 'joy'], 'safe': ['0.422', 'joy'], 'frisky': ['0.422', 'joy'], 'energy': ['0.422', 'joy'], 'calmness': ['0.422', 'joy'], 'symphony': ['0.422', 'joy'], 'helpful': ['0.422', 'joy'], 'musical': ['0.422', 'joy'], 'beginnings': ['0.422', 'joy'], 'nostress': ['0.421', 'joy'], 'soundness': ['0.421', 'joy'], 'promise': ['0.415', 'joy'], 'reunite': ['0.414', 'joy'], 'mother': ['0.412', 'joy'], 'salvation': ['0.412', 'joy'], 'poems': ['0.412', 'joy'], 'purify': ['0.409', 'joy'], 'travel': ['0.409', 'joy'], 'lavender': ['0.409', 'joy'], 'aromatherapy': ['0.409', 'joy'], 'inauguration': ['0.409', 'joy'], 'clown': ['0.409', 'joy'], 'immaculate': ['0.409', 'joy'], 'lighten': ['0.409', 'joy'], 'star': ['0.406', 'joy'], 'completing': ['0.406', 'joy'], 'heal': ['0.406', 'joy'], 'live': ['0.406', 'joy'], 'cash': ['0.406', 'joy'], 'companion': ['0.406', 'joy'], 'opportune': ['0.406', 'joy'], 'charity': ['0.406', 'joy'], 'flowers': ['0.406', 'joy'], 'wishing': ['0.406', 'joy'], 'income': ['0.403', 'joy'], 'soul': ['0.401', 'joy'], 'munchies': ['0.400', 'joy'], 'progress': ['0.397', 'joy'], 'indescribable': ['0.397', 'joy'], 'christian': ['0.397', 'joy'], 'emancipation': ['0.397', 'joy'], 'equality': ['0.394', 'joy'], 'rhythmical': ['0.394', 'joy'], 'childhood': ['0.394', 'joy'], 'calm': ['0.394', 'joy'], 'picnic': ['0.394', 'joy'], 'together': ['0.394', 'joy'], 'fullness': ['0.394', 'joy'], 'hammock': ['0.394', 'joy'], 'movies': ['0.393', 'joy'], 'zealous': ['0.393', 'joy'], 'choir': ['0.391', 'joy'], 'goofy': ['0.391', 'joy'], 'humanity': ['0.391', 'joy'], 'adventures': ['0.391', 'joy'], 'affluence': ['0.391', 'joy'], 'playground': ['0.391', 'joy'], 'starry': ['0.391', 'joy'], 'meaningful': ['0.391', 'joy'], 'auspicious': ['0.391', 'joy'], 'littlethings': ['0.391', 'joy'], 'warms': ['0.391', 'joy'], 'gesture': ['0.387', 'joy'], 'witty': ['0.382', 'joy'], 'shopping': ['0.382', 'joy'], 'vow': ['0.382', 'joy'], 'communion': ['0.379', 'joy'], 'jump': ['0.379', 'joy'], 'vibes': ['0.379', 'joy'], 'worship': ['0.379', 'joy'], 'reverend': ['0.377', 'joy'], 'unique': ['0.375', 'joy'], 'scenic': ['0.375', 'joy'], 'courtship': ['0.375', 'joy'], 'reunion': ['0.375', 'joy'], 'rising': ['0.375', 'joy'], 'full': ['0.375', 'joy'], 'redeemed': ['0.375', 'joy'], 'unforgettable': ['0.375', 'joy'], 'mirth': ['0.375', 'joy'], 'hymn': ['0.375', 'joy'], 'simplicity': ['0.375', 'joy'], 'spirits': ['0.375', 'joy'], 'youth': ['0.375', 'joy'], 'beaches': ['0.375', 'joy'], 'experience': ['0.375', 'joy'], 'advance': ['0.375', 'joy'], 'sonorous': ['0.375', 'joy'], 'baptismal': ['0.368', 'joy'], 'alliance': ['0.368', 'joy'], 'grant': ['0.366', 'joy'], 'moonlight': ['0.364', 'joy'], 'kitten': ['0.364', 'joy'], 'authentic': ['0.364', 'joy'], 'conciliation': ['0.364', 'joy'], 'sanctification': ['0.364', 'joy'], 'improve': ['0.364', 'joy'], 'pure': ['0.364', 'joy'], 'improves': ['0.364', 'joy'], 'weightloss': ['0.362', 'joy'], 'raspberries': ['0.360', 'joy'], 'feeling': ['0.359', 'joy'], 'devotional': ['0.359', 'joy'], 'fidelity': ['0.359', 'joy'], 'listenting': ['0.359', 'joy'], 'proficiency': ['0.359', 'joy'], 'jest': ['0.359', 'joy'], 'independent': ['0.359', 'joy'], 'tinsel': ['0.359', 'joy'], 'revival': ['0.359', 'joy'], 'sanctify': ['0.359', 'joy'], 'cocoa': ['0.353', 'joy'], 'giver': ['0.353', 'joy'], 'purpose': ['0.348', 'joy'], 'romp': ['0.348', 'joy'], 'deliverance': ['0.348', 'joy'], 'dolphin': ['0.348', 'joy'], 'unification': ['0.348', 'joy'], 'roaring': ['0.348', 'joy'], 'melody': ['0.348', 'joy'], 'choral': ['0.348', 'joy'], 'favor': ['0.348', 'joy'], 'exceed': ['0.348', 'joy'], 'hotyoga': ['0.347', 'joy'], 'electric': ['0.344', 'joy'], 'nowork': ['0.344', 'joy'], 'hedonism': ['0.344', 'joy'], 'pledge': ['0.344', 'joy'], 'humble': ['0.344', 'joy'], 'kiddos': ['0.344', 'joy'], 'thx': ['0.344', 'joy'], 'fruits': ['0.344', 'joy'], 'newstart': ['0.344', 'joy'], 'manicure': ['0.344', 'joy'], 'cookies': ['0.344', 'joy'], 'date': ['0.344', 'joy'], 'roadtrip': ['0.344', 'joy'], 'voluptuous': ['0.344', 'joy'], 'celebrity': ['0.342', 'joy'], 'rhythm': ['0.338', 'joy'], 'bridesmaid': ['0.338', 'joy'], 'obliging': ['0.333', 'joy'], 'familiarity': ['0.333', 'joy'], 'spa': ['0.333', 'joy'], 'connoisseur': ['0.333', 'joy'], 'coffee': ['0.333', 'joy'], 'edification': ['0.333', 'joy'], 'partner': ['0.333', 'joy'], 'garden': ['0.333', 'joy'], 'renovation': ['0.333', 'joy'], 'gazing': ['0.333', 'joy'], 'dawn': ['0.333', 'joy'], 'snowday': ['0.333', 'joy'], 'young': ['0.333', 'joy'], 'foodie': ['0.331', 'joy'], 'synchronize': ['0.329', 'joy'], 'saint': ['0.328', 'joy'], 'carol': ['0.328', 'joy'], 'hobby': ['0.328', 'joy'], 'noregrets': ['0.328', 'joy'], 'amnesty': ['0.328', 'joy'], 'healing': ['0.328', 'joy'], 'tribulation': ['0.328', 'joy'], 'TRUE': ['0.328', 'joy'], 'chirping': ['0.328', 'joy'], 'psalm': ['0.328', 'joy'], 'pedicure': ['0.328', 'joy'], 'respite': ['0.324', 'joy'], 'mellow': ['0.318', 'joy'], 'recreational': ['0.318', 'joy'], 'classics': ['0.318', 'joy'], 'cousins': ['0.318', 'joy'], 'restorative': ['0.318', 'joy'], 'lazyday': ['0.318', 'joy'], 'reconciliation': ['0.316', 'joy'], 'superman': ['0.312', 'joy'], 'living': ['0.312', 'joy'], 'simplify': ['0.312', 'joy'], 'recovery': ['0.312', 'joy'], 'relight': ['0.312', 'joy'], 'sunset': ['0.312', 'joy'], 'crafts': ['0.312', 'joy'], 'pony': ['0.312', 'joy'], 'deal': ['0.312', 'joy'], 'presto': ['0.312', 'joy'], 'fitness': ['0.312', 'joy'], 'sterling': ['0.312', 'joy'], 'wisdom': ['0.312', 'joy'], 'dove': ['0.312', 'joy'], 'playhouse': ['0.312', 'joy'], 'woods': ['0.312', 'joy'], 'muchneeded': ['0.312', 'joy'], 'progression': ['0.312', 'joy'], 'lord': ['0.312', 'joy'], 'improvement': ['0.309', 'joy'], 'infant': ['0.309', 'joy'], 'absolution': ['0.306', 'joy'], 'endless': ['0.303', 'joy'], 'established': ['0.303', 'joy'], 'banquet': ['0.303', 'joy'], 'decorating': ['0.303', 'joy'], 'honest': ['0.303', 'joy'], 'gently': ['0.303', 'joy'], 'providing': ['0.300', 'joy'], 'softly': ['0.297', 'joy'], 'yearning': ['0.297', 'joy'], 'spiritual': ['0.297', 'joy'], 'growth': ['0.297', 'joy'], 'quaint': ['0.297', 'joy'], 'notable': ['0.297', 'joy'], 'feat': ['0.297', 'joy'], 'strolling': ['0.297', 'joy'], 'lounging': ['0.297', 'joy'], 'meditating': ['0.297', 'joy'], 'fraternal': ['0.297', 'joy'], 'joker': ['0.297', 'joy'], 'spending': ['0.297', 'joy'], 'light': ['0.297', 'joy'], 'morals': ['0.297', 'joy'], 'mine': ['0.297', 'joy'], 'harvest': ['0.297', 'joy'], 'breeze': ['0.297', 'joy'], 'fervor': ['0.295', 'joy'], 'volunteer': ['0.294', 'joy'], 'quotes': ['0.294', 'joy'], 'gain': ['0.288', 'joy'], 'create': ['0.288', 'joy'], 'midwife': ['0.288', 'joy'], 'resources': ['0.288', 'joy'], 'psalms': ['0.288', 'joy'], 'taoism': ['0.288', 'joy'], 'hippie': ['0.288', 'joy'], 'receiving': ['0.287', 'joy'], 'ocean': ['0.286', 'joy'], 'lsd': ['0.286', 'joy'], 'snowfall': ['0.286', 'joy'], 'hiking': ['0.283', 'joy'], 'chill': ['0.281', 'joy'], 'evergreen': ['0.281', 'joy'], 'candid': ['0.281', 'joy'], 'unwind': ['0.281', 'joy'], 'toast': ['0.281', 'joy'], 'destination': ['0.281', 'joy'], 'infinity': ['0.281', 'joy'], 'buddhist': ['0.281', 'joy'], 'rested': ['0.281', 'joy'], 'truce': ['0.281', 'joy'], 'buzzing': ['0.281', 'joy'], 'venerable': ['0.279', 'joy'], 'retirement': ['0.275', 'joy'], 'balance': ['0.275', 'joy'], 'buddha': ['0.273', 'joy'], 'drinks': ['0.273', 'joy'], 'compensate': ['0.273', 'joy'], 'fireplace': ['0.273', 'joy'], 'devout': ['0.273', 'joy'], 'rapt': ['0.273', 'joy'], 'swim': ['0.273', 'joy'], 'snowing': ['0.271', 'joy'], 'sublimation': ['0.270', 'joy'], 'fruity': ['0.268', 'joy'], 'whim': ['0.266', 'joy'], 'salute': ['0.266', 'joy'], 'sun': ['0.266', 'joy'], 'sunday': ['0.266', 'joy'], 'buddhism': ['0.266', 'joy'], 'cathedral': ['0.266', 'joy'], 'ministry': ['0.266', 'joy'], 'humility': ['0.266', 'joy'], 'yoga': ['0.266', 'joy'], 'fortitude': ['0.266', 'joy'], 'sketching': ['0.266', 'joy'], 'paragon': ['0.266', 'joy'], 'waves': ['0.266', 'joy'], 'artwork': ['0.265', 'joy'], 'practiced': ['0.264', 'joy'], 'clean': ['0.260', 'joy'], 'food': ['0.258', 'joy'], 'exercise': ['0.258', 'joy'], 'immerse': ['0.258', 'joy'], 'running': ['0.258', 'joy'], 'fitting': ['0.258', 'joy'], 'found': ['0.258', 'joy'], 'clarity': ['0.258', 'joy'], 'autumn': ['0.254', 'joy'], 'countryside': ['0.250', 'joy'], 'undying': ['0.250', 'joy'], 'diary': ['0.250', 'joy'], 'football': ['0.250', 'joy'], 'fit': ['0.250', 'joy'], 'visitor': ['0.250', 'joy'], 'ardent': ['0.250', 'joy'], 'chilled': ['0.250', 'joy'], 'chirp': ['0.250', 'joy'], 'mountain': ['0.250', 'joy'], 'expedient': ['0.250', 'joy'], 'hardy': ['0.250', 'joy'], 'candles': ['0.250', 'joy'], 'contagious': ['0.250', 'joy'], 'advocacy': ['0.250', 'joy'], 'outdoors': ['0.250', 'joy'], 'clap': ['0.250', 'joy'], 'demonstrative': ['0.250', 'joy'], 'morning': ['0.250', 'joy'], 'cradle': ['0.247', 'joy'], 'trance': ['0.246', 'joy'], 'preservative': ['0.242', 'joy'], 'procession': ['0.242', 'joy'], 'grow': ['0.242', 'joy'], 'intense': ['0.242', 'joy'], 'church': ['0.242', 'joy'], 'pay': ['0.242', 'joy'], 'breakfast': ['0.242', 'joy'], 'supremacy': ['0.242', 'joy'], 'consecration': ['0.235', 'joy'], 'incense': ['0.234', 'joy'], 'ordination': ['0.234', 'joy'], 'lyre': ['0.234', 'joy'], 'vote': ['0.234', 'joy'], 'nursery': ['0.234', 'joy'], 'skiing': ['0.234', 'joy'], 'humbled': ['0.234', 'joy'], 'salary': ['0.234', 'joy'], 'art': ['0.234', 'joy'], 'candle': ['0.234', 'joy'], 'snowy': ['0.234', 'joy'], 'doll': ['0.234', 'joy'], 'lifetime': ['0.234', 'joy'], 'pastry': ['0.229', 'joy'], 'remedy': ['0.227', 'joy'], 'firefly': ['0.227', 'joy'], 'chilling': ['0.227', 'joy'], 'rescue': ['0.225', 'joy'], 'perspective': ['0.221', 'joy'], 'orchard': ['0.221', 'joy'], 'camping': ['0.221', 'joy'], 'orchestra': ['0.221', 'joy'], 'breezy': ['0.219', 'joy'], 'possess': ['0.219', 'joy'], 'lamb': ['0.219', 'joy'], 'castle': ['0.219', 'joy'], 'gardens': ['0.219', 'joy'], 'brisk': ['0.212', 'joy'], 'ribbon': ['0.212', 'joy'], 'raindrops': ['0.212', 'joy'], 'reggae': ['0.212', 'joy'], 'pho': ['0.212', 'joy'], 'teach': ['0.212', 'joy'], 'repay': ['0.212', 'joy'], 'listneing': ['0.212', 'joy'], 'mucis': ['0.212', 'joy'], 'bouttime': ['0.212', 'joy'], 'countrymusic': ['0.212', 'joy'], 'nap': ['0.206', 'joy'], 'vernal': ['0.203', 'joy'], 'lights': ['0.203', 'joy'], 'legalized': ['0.203', 'joy'], 'hire': ['0.203', 'joy'], 'uncontrollable': ['0.203', 'joy'], 'chant': ['0.203', 'joy'], 'unexpected': ['0.203', 'joy'], 'stillness': ['0.203', 'joy'], 'cove': ['0.203', 'joy'], 'save': ['0.200', 'joy'], 'symmetry': ['0.197', 'joy'], 'pastor': ['0.197', 'joy'], 'atone': ['0.197', 'joy'], 'trees': ['0.197', 'joy'], 'oneday': ['0.197', 'joy'], 'scripture': ['0.196', 'joy'], 'mountains': ['0.194', 'joy'], 'quiet': ['0.188', 'joy'], 'liquor': ['0.188', 'joy'], 'walking': ['0.188', 'joy'], 'craziness': ['0.188', 'joy'], 'custom': ['0.188', 'joy'], 'pathway': ['0.188', 'joy'], 'forefathers': ['0.188', 'joy'], 'sympathetic': ['0.188', 'joy'], 'whiteness': ['0.188', 'joy'], 'tea': ['0.188', 'joy'], 'soppy': ['0.188', 'joy'], 'patient': ['0.186', 'joy'], 'dollhouse': ['0.182', 'joy'], 'bath': ['0.182', 'joy'], 'score': ['0.182', 'joy'], 'movingon': ['0.182', 'joy'], 'supporter': ['0.180', 'joy'], 'accompaniment': ['0.179', 'joy'], 'pursuit': ['0.176', 'joy'], 'outburst': ['0.176', 'joy'], 'frosty': ['0.174', 'joy'], 'workout': ['0.172', 'joy'], 'closure': ['0.172', 'joy'], 'luncheon': ['0.172', 'joy'], 'wintery': ['0.172', 'joy'], 'service': ['0.172', 'joy'], 'civilized': ['0.169', 'joy'], 'wages': ['0.169', 'joy'], 'december': ['0.167', 'joy'], 'fain': ['0.167', 'joy'], 'glide': ['0.162', 'joy'], 'acrobat': ['0.162', 'joy'], 'finally': ['0.157', 'joy'], 'chai': ['0.156', 'joy'], 'obtainable': ['0.156', 'joy'], 'organization': ['0.156', 'joy'], 'peppermint': ['0.156', 'joy'], 'stroll': ['0.156', 'joy'], 'break': ['0.156', 'joy'], 'elf': ['0.156', 'joy'], 'bathtub': ['0.156', 'joy'], 'reproductive': ['0.156', 'joy'], 'balm': ['0.152', 'joy'], 'advent': ['0.152', 'joy'], 'measured': ['0.152', 'joy'], 'scifi': ['0.152', 'joy'], 'spaniel': ['0.145', 'joy'], 'tan': ['0.141', 'joy'], 'ditty': ['0.141', 'joy'], 'bubble': ['0.141', 'joy'], 'beer': ['0.141', 'joy'], 'simple': ['0.141', 'joy'], 'oils': ['0.141', 'joy'], 'green': ['0.137', 'joy'], 'books': ['0.136', 'joy'], 'buss': ['0.136', 'joy'], 'makingdisciples': ['0.136', 'joy'], 'chow': ['0.135', 'joy'], 'pitter': ['0.134', 'joy'], 'flows': ['0.130', 'joy'], 'silence': ['0.127', 'joy'], 'bookstore': ['0.125', 'joy'], 'circumstances': ['0.125', 'joy'], 'solitude': ['0.125', 'joy'], 'roadster': ['0.125', 'joy'], 'wine': ['0.125', 'joy'], 'wilderness': ['0.121', 'joy'], 'soak': ['0.121', 'joy'], 'priesthood': ['0.121', 'joy'], 'japan': ['0.121', 'joy'], 'critical': ['0.116', 'joy'], 'neutral': ['0.114', 'joy'], 'wind': ['0.109', 'joy'], 'hunting': ['0.109', 'joy'], 'untie': ['0.109', 'joy'], 'opera': ['0.109', 'joy'], 'white': ['0.109', 'joy'], 'weight': ['0.109', 'joy'], 'sand': ['0.109', 'joy'], 'classical': ['0.106', 'joy'], 'labor': ['0.106', 'joy'], 'affliction': ['0.103', 'joy'], 'lake': ['0.103', 'joy'], 'organ': ['0.094', 'joy'], 'dwelling': ['0.094', 'joy'], 'tree': ['0.090', 'joy'], 'pond': ['0.089', 'joy'], 'latte': ['0.078', 'joy'], 'marrow': ['0.078', 'joy'], 'sipping': ['0.076', 'joy'], 'benign': ['0.074', 'joy'], 'majority': ['0.073', 'joy'], 'leaf': ['0.065', 'joy'], 'troubles': ['0.062', 'joy'], 'basketball': ['0.062', 'joy'], 'explosions': ['0.061', 'joy'], 'cream': ['0.061', 'joy'], 'shepherd': ['0.058', 'joy'], 'tuesday': ['0.047', 'joy'], 'turbulence': ['0.045', 'joy'], 'calf': ['0.040', 'joy'], 'hardship': ['0.031', 'joy'], 'unhappiness': ['0.016', 'joy'], 'sixty': ['0.016', 'joy']}
diction={'heartbreaking': ['0.969', 'sadness'],'ugly': ['0.609', 'sadness'], 'mourning': ['0.969', 'sadness'], 'tragic': ['0.961', 'sadness'], 'holocaust': ['0.953', 'sadness'], 'suicidal': ['0.941', 'sadness'], 'misery': ['0.938', 'sadness'], 'massacre': ['0.931', 'sadness'], 'euthanasia': ['0.927', 'sadness'], 'depression': ['0.925', 'sadness'], 'fatal': ['0.922', 'sadness'], 'bereavement': ['0.922', 'sadness'], 'grieving': ['0.922', 'sadness'], 'bereaved': ['0.920', 'sadness'], 'devastation': ['0.917', 'sadness'], 'death': ['0.915', 'sadness'], 'suicide': ['0.912', 'sadness'], 'devastated': ['0.912', 'sadness'], 'catastrophe': ['0.911', 'sadness'], 'horrifying': ['0.907', 'sadness'], 'tragedy': ['0.906', 'sadness'], 'died': ['0.906', 'sadness'], 'depressing': ['0.906', 'sadness'], 'anguish': ['0.902', 'sadness'], 'agony': ['0.900', 'sadness'], 'deadly': ['0.898', 'sadness'], 'weeping': ['0.896', 'sadness'], 'deceased': ['0.891', 'sadness'], 'stillbirth': ['0.891', 'sadness'], 'murderer': ['0.877', 'sadness'], 'cancer': ['0.875', 'sadness'], 'dying': ['0.875', 'sadness'], 'rape': ['0.875', 'sadness'], 'devastating': ['0.875', 'sadness'], 'sadness': ['0.864', 'sadness'], 'morbidity': ['0.864', 'sadness'], 'execution': ['0.859', 'sadness'], 'abandonment': ['0.859', 'sadness'], 'crucifixion': ['0.859', 'sadness'], 'grief': ['0.859', 'sadness'], 'depressed': ['0.859', 'sadness'], 'perish': ['0.859', 'sadness'], 'traumatic': ['0.859', 'sadness'], 'atrocity': ['0.859', 'sadness'], 'cadaver': ['0.853', 'sadness'], 'betrayed': ['0.848', 'sadness'], 'treachery': ['0.848', 'sadness'], 'funeral': ['0.844', 'sadness'], 'grieve': ['0.844', 'sadness'], 'murderous': ['0.844', 'sadness'], 'miserable': ['0.844', 'sadness'], 'hopelessness': ['0.844', 'sadness'], 'persecution': ['0.844', 'sadness'], 'sad': ['0.844', 'sadness'], 'suffering': ['0.844', 'sadness'], 'sorrow': ['0.844', 'sadness'], 'homicide': ['0.844', 'sadness'], 'slaughtering': ['0.844', 'sadness'], 'destroyed': ['0.844', 'sadness'], 'horrific': ['0.844', 'sadness'], 'unhappiness': ['0.839', 'sadness'], 'crippled': ['0.836', 'sadness'], 'bloodshed': ['0.836', 'sadness'], 'pained': ['0.833', 'sadness'], 'manslaughter': ['0.833', 'sadness'], 'carnage': ['0.833', 'sadness'], 'unbearable': ['0.830', 'sadness'], 'stillborn': ['0.830', 'sadness'], 'torment': ['0.828', 'sadness'], 'helplessness': ['0.828', 'sadness'], 'annihilation': ['0.828', 'sadness'], 'slavery': ['0.828', 'sadness'], 'annihilated': ['0.828', 'sadness'], 'enslaved': ['0.828', 'sadness'], 'casualty': ['0.828', 'sadness'], 'horrors': ['0.828', 'sadness'], 'murder': ['0.828', 'sadness'], 'mourn': ['0.828', 'sadness'], 'morbid': ['0.828', 'sadness'], 'abandoned': ['0.828', 'sadness'], 'sickness': ['0.828', 'sadness'], 'mutilation': ['0.828', 'sadness'], 'miscarriage': ['0.824', 'sadness'], 'starvation': ['0.819', 'sadness'], 'cruelty': ['0.812', 'sadness'], 'childloss': ['0.812', 'sadness'], 'disgrace': ['0.812', 'sadness'], 'killing': ['0.812', 'sadness'], 'oppression': ['0.812', 'sadness'], 'terrorism': ['0.812', 'sadness'], 'failure': ['0.812', 'sadness'], 'famine': ['0.812', 'sadness'], 'heartache': ['0.812', 'sadness'], 'burial': ['0.812', 'sadness'], 'saddens': ['0.812', 'sadness'], 'distraught': ['0.812', 'sadness'], 'despair': ['0.812', 'sadness'], 'sadly': ['0.812', 'sadness'], 'mournful': ['0.812', 'sadness'], 'bloody': ['0.806', 'sadness'], 'inhumanity': ['0.804', 'sadness'], 'perishing': ['0.804', 'sadness'], 'malignancy': ['0.803', 'sadness'], 'mortification': ['0.802', 'sadness'], 'kill': ['0.797', 'sadness'], 'lifeless': ['0.797', 'sadness'], 'dreadful': ['0.797', 'sadness'], 'slave': ['0.797', 'sadness'], 'desolation': ['0.797', 'sadness'], 'devastate': ['0.797', 'sadness'], 'perished': ['0.797', 'sadness'], 'assassination': ['0.797', 'sadness'], 'mortuary': ['0.797', 'sadness'], 'dreadfully': ['0.797', 'sadness'], 'leukemia': ['0.797', 'sadness'], 'sarcoma': ['0.797', 'sadness'], 'lethal': ['0.797', 'sadness'], 'gallows': ['0.797', 'sadness'], 'brokenheart': ['0.792', 'sadness'], 'banishment': ['0.790', 'sadness'], 'afflict': ['0.789', 'sadness'], 'disheartened': ['0.788', 'sadness'], 'bury': ['0.781', 'sadness'], 'desecration': ['0.781', 'sadness'], 'demoralized': ['0.781', 'sadness'], 'tumour': ['0.781', 'sadness'], 'terrorize': ['0.781', 'sadness'], 'crying': ['0.781', 'sadness'], 'heartbreak': ['0.781', 'sadness'], 'die': ['0.773', 'sadness'], 'lynch': ['0.773', 'sadness'], 'sufferer': ['0.770', 'sadness'], 'loneliness': ['0.766', 'sadness'], 'abortion': ['0.766', 'sadness'], 'dismemberment': ['0.766', 'sadness'], 'diseased': ['0.766', 'sadness'], 'fearful': ['0.766', 'sadness'], 'destitute': ['0.766', 'sadness'], 'torture': ['0.766', 'sadness'], 'slayer': ['0.766', 'sadness'], 'cemetery': ['0.766', 'sadness'], 'fatality': ['0.766', 'sadness'], 'condolence': ['0.766', 'sadness'], 'doomed': ['0.766', 'sadness'], 'painfully': ['0.758', 'sadness'], 'moribund': ['0.758', 'sadness'], 'disaster': ['0.758', 'sadness'], 'depress': ['0.755', 'sadness'], 'condemnation': ['0.754', 'sadness'], 'victimized': ['0.750', 'sadness'], 'obliteration': ['0.750', 'sadness'], 'depressive': ['0.750', 'sadness'], 'terrorist': ['0.750', 'sadness'], 'guilt': ['0.750', 'sadness'], 'incest': ['0.750', 'sadness'], 'pandemic': ['0.750', 'sadness'], 'unhappy': ['0.750', 'sadness'], 'defeated': ['0.750', 'sadness'], 'painful': ['0.750', 'sadness'], 'deplorable': ['0.750', 'sadness'], 'damnation': ['0.750', 'sadness'], 'doomsday': ['0.750', 'sadness'], 'corpse': ['0.750', 'sadness'], 'abduction': ['0.750', 'sadness'], 'sorrowful': ['0.750', 'sadness'], 'regretful': ['0.750', 'sadness'], 'desperation': ['0.750', 'sadness'], 'cry': ['0.750', 'sadness'], 'sickening': ['0.750', 'sadness'], 'hemorrhage': ['0.750', 'sadness'], 'unfairness': ['0.745', 'sadness'], 'molestation': ['0.744', 'sadness'], 'exile': ['0.742', 'sadness'], 'abysmal': ['0.742', 'sadness'], 'hellish': ['0.738', 'sadness'], 'exterminate': ['0.736', 'sadness'], 'disgraced': ['0.734', 'sadness'], 'homeless': ['0.734', 'sadness'], 'destroying': ['0.734', 'sadness'], 'battered': ['0.734', 'sadness'], 'betrayal': ['0.734', 'sadness'], 'horrid': ['0.734', 'sadness'], 'warfare': ['0.734', 'sadness'], 'assassin': ['0.734', 'sadness'], 'disastrous': ['0.734', 'sadness'], 'lonesome': ['0.734', 'sadness'], 'miserably': ['0.734', 'sadness'], 'morgue': ['0.734', 'sadness'], 'slaughter': ['0.734', 'sadness'], 'earthquake': ['0.734', 'sadness'], 'orphan': ['0.734', 'sadness'], 'listless': ['0.729', 'sadness'], 'grave': ['0.727', 'sadness'], 'emptiness': ['0.727', 'sadness'], 'unfortunately': ['0.727', 'sadness'], 'alienated': ['0.727', 'sadness'], 'fraught': ['0.722', 'sadness'], 'forsaken': ['0.719', 'sadness'], 'leprosy': ['0.719', 'sadness'], 'cried': ['0.719', 'sadness'], 'paralysis': ['0.719', 'sadness'], 'malicious': ['0.719', 'sadness'], 'ashamed': ['0.719', 'sadness'], 'woe': ['0.719', 'sadness'], 'danger': ['0.719', 'sadness'], 'disheartening': ['0.719', 'sadness'], 'heartless': ['0.719', 'sadness'], 'violently': ['0.719', 'sadness'], 'cripple': ['0.719', 'sadness'], 'horror': ['0.719', 'sadness'], 'atrophy': ['0.719', 'sadness'], 'missing': ['0.719', 'sadness'], 'emaciated': ['0.719', 'sadness'], 'pain': ['0.719', 'sadness'], 'demise': ['0.717', 'sadness'], 'sickly': ['0.712', 'sadness'], 'disgruntled': ['0.712', 'sadness'], 'violence': ['0.712', 'sadness'], 'rejected': ['0.712', 'sadness'], 'torn': ['0.710', 'sadness'], 'calamity': ['0.709', 'sadness'], 'grim': ['0.708', 'sadness'], 'grievous': ['0.704', 'sadness'], 'hearse': ['0.703', 'sadness'], 'extinct': ['0.703', 'sadness'], 'crushed': ['0.703', 'sadness'], 'isolation': ['0.703', 'sadness'], 'meltdown': ['0.703', 'sadness'], 'obit': ['0.703', 'sadness'], 'paralyzed': ['0.703', 'sadness'], 'carcinoma': ['0.703', 'sadness'], 'suffocating': ['0.703', 'sadness'], 'deformed': ['0.703', 'sadness'], 'inhuman': ['0.703', 'sadness'], 'punishing': ['0.703', 'sadness'], 'incurable': ['0.703', 'sadness'], 'strangle': ['0.703', 'sadness'], 'disfigured': ['0.703', 'sadness'], 'victim': ['0.703', 'sadness'], 'deformity': ['0.703', 'sadness'], 'slaughterhouse': ['0.703', 'sadness'], 'decomposition': ['0.703', 'sadness'], 'humiliate': ['0.703', 'sadness'], 'buried': ['0.703', 'sadness'], 'oppressor': ['0.703', 'sadness'], 'abandon': ['0.703', 'sadness'], 'tearful': ['0.703', 'sadness'], 'isolate': ['0.703', 'sadness'], 'lifesucks': ['0.700', 'sadness'], 'hell': ['0.700', 'sadness'], 'ruinous': ['0.698', 'sadness'], 'banish': ['0.697', 'sadness'], 'ruined': ['0.697', 'sadness'], 'accursed': ['0.697', 'sadness'], 'widow': ['0.697', 'sadness'], 'vanished': ['0.695', 'sadness'], 'displaced': ['0.691', 'sadness'], 'poverty': ['0.690', 'sadness'], 'illness': ['0.688', 'sadness'], 'hopeless': ['0.688', 'sadness'], 'travesty': ['0.688', 'sadness'], 'deserted': ['0.688', 'sadness'], 'regretting': ['0.688', 'sadness'], 'loss': ['0.688', 'sadness'], 'pathetic': ['0.688', 'sadness'], 'nohope': ['0.688', 'sadness'], 'stab': ['0.688', 'sadness'], 'shooting': ['0.688', 'sadness'], 'foreveralone': ['0.688', 'sadness'], 'imprisoned': ['0.688', 'sadness'], 'insanity': ['0.688', 'sadness'], 'hurtful': ['0.688', 'sadness'], 'terminal': ['0.688', 'sadness'], 'epidemic': ['0.688', 'sadness'], 'hurt': ['0.688', 'sadness'], 'depraved': ['0.688', 'sadness'], 'banished': ['0.688', 'sadness'], 'infidelity': ['0.688', 'sadness'], 'neglected': ['0.688', 'sadness'], 'sob': ['0.688', 'sadness'], 'teary': ['0.688', 'sadness'], 'dementia': ['0.688', 'sadness'], 'widower': ['0.688', 'sadness'], 'hospice': ['0.688', 'sadness'], 'dismissal': ['0.686', 'sadness'], 'alienation': ['0.685', 'sadness'], 'hardship': ['0.685', 'sadness'], 'kidnap': ['0.682', 'sadness'], 'choke': ['0.682', 'sadness'], 'bleeding': ['0.673', 'sadness'], 'outcast': ['0.672', 'sadness'], 'despairing': ['0.672', 'sadness'], 'woefully': ['0.672', 'sadness'], 'belittle': ['0.672', 'sadness'], 'evil': ['0.672', 'sadness'], 'disparage': ['0.672', 'sadness'], 'feelingdown': ['0.672', 'sadness'], 'imprisonment': ['0.672', 'sadness'], 'frightful': ['0.672', 'sadness'], 'punished': ['0.672', 'sadness'], 'missingyou': ['0.672', 'sadness'], 'wretched': ['0.672', 'sadness'], 'abortive': ['0.672', 'sadness'], 'obituary': ['0.672', 'sadness'], 'gory': ['0.672', 'sadness'], 'wretch': ['0.672', 'sadness'], 'poison': ['0.672', 'sadness'], 'coffin': ['0.672', 'sadness'], 'deprivation': ['0.672', 'sadness'], 'malevolent': ['0.672', 'sadness'], 'wail': ['0.672', 'sadness'], 'disabled': ['0.672', 'sadness'], 'decomposed': ['0.672', 'sadness'], 'barren': ['0.670', 'sadness'], 'poisoned': ['0.667', 'sadness'], 'executioner': ['0.667', 'sadness'], 'disease': ['0.665', 'sadness'], 'oppress': ['0.664', 'sadness'], 'disembodied': ['0.660', 'sadness'], 'tear': ['0.656', 'sadness'], 'hate': ['0.656', 'sadness'], 'lonely': ['0.656', 'sadness'], 'dreary': ['0.656', 'sadness'], 'blighted': ['0.656', 'sadness'], 'ailing': ['0.656', 'sadness'], 'demonic': ['0.656', 'sadness'], 'peril': ['0.656', 'sadness'], 'jail': ['0.656', 'sadness'], 'lamenting': ['0.656', 'sadness'], 'shitty': ['0.656', 'sadness'], 'polio': ['0.656', 'sadness'], 'mangle': ['0.656', 'sadness'], 'ruin': ['0.656', 'sadness'], 'weep': ['0.656', 'sadness'], 'steal': ['0.656', 'sadness'], 'casket': ['0.656', 'sadness'], 'bleak': ['0.656', 'sadness'], 'carcass': ['0.653', 'sadness'], 'regretted': ['0.652', 'sadness'], 'beating': ['0.652', 'sadness'], 'cowardice': ['0.652', 'sadness'], 'disability': ['0.648', 'sadness'], 'affliction': ['0.645', 'sadness'], 'emergency': ['0.641', 'sadness'], 'hatred': ['0.641', 'sadness'], 'termination': ['0.641', 'sadness'], 'awful': ['0.641', 'sadness'], 'exorcism': ['0.641', 'sadness'], 'sinful': ['0.641', 'sadness'], 'scourge': ['0.641', 'sadness'], 'perilous': ['0.641', 'sadness'], 'poisonous': ['0.641', 'sadness'], 'worry': ['0.641', 'sadness'], 'drown': ['0.641', 'sadness'], 'infertility': ['0.641', 'sadness'], 'shroud': ['0.641', 'sadness'], 'powerless': ['0.641', 'sadness'], 'woeful': ['0.641', 'sadness'], 'failing': ['0.641', 'sadness'], 'terribly': ['0.641', 'sadness'], 'inequality': ['0.641', 'sadness'], 'incarceration': ['0.641', 'sadness'], 'stricken': ['0.641', 'sadness'], 'psychosis': ['0.638', 'sadness'], 'disappointed': ['0.636', 'sadness'], 'demolish': ['0.636', 'sadness'], 'dismay': ['0.636', 'sadness'], 'lament': ['0.636', 'sadness'], 'burdensome': ['0.634', 'sadness'], 'mausoleum': ['0.630', 'sadness'], 'shattered': ['0.630', 'sadness'], 'tyrant': ['0.625', 'sadness'], 'disappointing': ['0.625', 'sadness'], 'insurmountable': ['0.625', 'sadness'], 'wound': ['0.625', 'sadness'], 'impotence': ['0.625', 'sadness'], 'wrecked': ['0.625', 'sadness'], 'abuse': ['0.625', 'sadness'], 'demolished': ['0.625', 'sadness'], 'palsy': ['0.625', 'sadness'], 'lost': ['0.625', 'sadness'], 'posthumous': ['0.625', 'sadness'], 'gloom': ['0.625', 'sadness'], 'schizophrenia': ['0.625', 'sadness'], 'cursed': ['0.625', 'sadness'], 'undesired': ['0.625', 'sadness'], 'forlorn': ['0.625', 'sadness'], 'terminate': ['0.625', 'sadness'], 'dishonor': ['0.625', 'sadness'], 'regret': ['0.625', 'sadness'], 'bitterly': ['0.625', 'sadness'], 'hurting': ['0.625', 'sadness'], 'duress': ['0.625', 'sadness'], 'oppressive': ['0.625', 'sadness'], 'deteriorate': ['0.625', 'sadness'], 'soulless': ['0.623', 'sadness'], 'divorce': ['0.623', 'sadness'], 'melancholy': ['0.621', 'sadness'], 'cremation': ['0.621', 'sadness'], 'bomb': ['0.621', 'sadness'], 'forsake': ['0.621', 'sadness'], 'worried': ['0.621', 'sadness'], 'plight': ['0.621', 'sadness'], 'unforgiving': ['0.612', 'sadness'], 'sepsis': ['0.611', 'sadness'], 'overwhelmed': ['0.609', 'sadness'], 'fearfully': ['0.609', 'sadness'], 'languishing': ['0.609', 'sadness'], 'alcoholism': ['0.609', 'sadness'], 'irreparable': ['0.609', 'sadness'], 'bankrupt': ['0.609', 'sadness'], 'gore': ['0.609', 'sadness'], 'debacle': ['0.609', 'sadness'], 'cruel': ['0.609', 'sadness'], 'injured': ['0.609', 'sadness'], 'faithless': ['0.609', 'sadness'], 'ugliness': ['0.609', 'sadness'], 'derogatory': ['0.609', 'sadness'], 'injure': ['0.609', 'sadness'], 'disappoint': ['0.609', 'sadness'], 'crushing': ['0.609', 'sadness'], 'shackle': ['0.609', 'sadness'], 'dire': ['0.609', 'sadness'], 'sacrifices': ['0.609', 'sadness'], 'breakup': ['0.609', 'sadness'], 'subjugation': ['0.609', 'sadness'], 'excluded': ['0.609', 'sadness'], 'sinner': ['0.609', 'sadness'], 'degrading': ['0.609', 'sadness'], 'worthless': ['0.609', 'sadness'], 'guilty': ['0.609', 'sadness'], 'shatter': ['0.609', 'sadness'], 'disparaging': ['0.609', 'sadness'], 'dilapidated': ['0.609', 'sadness'], 'shameful': ['0.609', 'sadness'], 'curse': ['0.608', 'sadness'], 'anthrax': ['0.600', 'sadness'], 'robbery': ['0.600', 'sadness'], 'alone': ['0.600', 'sadness'], 'angst': ['0.598', 'sadness'], 'losing': ['0.594', 'sadness'], 'prison': ['0.594', 'sadness'], 'somber': ['0.594', 'sadness'], 'contaminated': ['0.594', 'sadness'], 'deprived': ['0.594', 'sadness'], 'martyrdom': ['0.594', 'sadness'], 'irreconcilable': ['0.594', 'sadness'], 'poaching': ['0.594', 'sadness'], 'bawl': ['0.594', 'sadness'], 'eviction': ['0.594', 'sadness'], 'ill': ['0.594', 'sadness'], 'helpless': ['0.594', 'sadness'], 'downfall': ['0.594', 'sadness'], 'deportation': ['0.594', 'sadness'], 'crumbling': ['0.594', 'sadness'], 'distress': ['0.594', 'sadness'], 'disappointment': ['0.594', 'sadness'], 'demon': ['0.594', 'sadness'], 'nothingness': ['0.594', 'sadness'], 'condolences': ['0.594', 'sadness'], 'crypt': ['0.594', 'sadness'], 'longing': ['0.594', 'sadness'], 'shame': ['0.594', 'sadness'], 'captivity': ['0.594', 'sadness'], 'obliterate': ['0.594', 'sadness'], 'chaos': ['0.594', 'sadness'], 'violation': ['0.594', 'sadness'], 'vendetta': ['0.594', 'sadness'], 'broken': ['0.594', 'sadness'], 'abyss': ['0.594', 'sadness'], 'petloss': ['0.594', 'sadness'], 'offender': ['0.594', 'sadness'], 'remorse': ['0.594', 'sadness'], 'betray': ['0.594', 'sadness'], 'dysentery': ['0.593', 'sadness'], 'blight': ['0.591', 'sadness'], 'melancholic': ['0.591', 'sadness'], 'rupture': ['0.588', 'sadness'], 'traitor': ['0.588', 'sadness'], 'haggard': ['0.587', 'sadness'], 'lie': ['0.585', 'sadness'], 'cholera': ['0.583', 'sadness'], 'degeneracy': ['0.578', 'sadness'], 'undesirable': ['0.578', 'sadness'], 'gloomy': ['0.578', 'sadness'], 'turmoil': ['0.578', 'sadness'], 'terrible': ['0.578', 'sadness'], 'frighten': ['0.578', 'sadness'], 'unwell': ['0.578', 'sadness'], 'bully': ['0.578', 'sadness'], 'bitterness': ['0.578', 'sadness'], 'discrimination': ['0.578', 'sadness'], 'whine': ['0.578', 'sadness'], 'humiliation': ['0.578', 'sadness'], 'sordid': ['0.578', 'sadness'], 'immoral': ['0.578', 'sadness'], 'harmful': ['0.578', 'sadness'], 'interment': ['0.578', 'sadness'], 'denied': ['0.578', 'sadness'], 'damage': ['0.578', 'sadness'], 'delirium': ['0.576', 'sadness'], 'captive': ['0.576', 'sadness'], 'pessimism': ['0.576', 'sadness'], 'deplore': ['0.576', 'sadness'], 'whimper': ['0.576', 'sadness'], 'disliked': ['0.576', 'sadness'], 'devil': ['0.576', 'sadness'], 'damages': ['0.576', 'sadness'], 'hateful': ['0.575', 'sadness'], 'bigoted': ['0.574', 'sadness'], 'perdition': ['0.569', 'sadness'], 'adultery': ['0.566', 'sadness'], 'corrupting': ['0.565', 'sadness'], 'worsening': ['0.562', 'sadness'], 'flog': ['0.562', 'sadness'], 'dismal': ['0.562', 'sadness'], 'comatose': ['0.562', 'sadness'], 'autopsy': ['0.562', 'sadness'], 'worrying': ['0.562', 'sadness'], 'deceive': ['0.562', 'sadness'], 'tomb': ['0.562', 'sadness'], 'deceit': ['0.562', 'sadness'], 'wallow': ['0.562', 'sadness'], 'pessimist': ['0.562', 'sadness'], 'rejection': ['0.562', 'sadness'], 'sadday': ['0.562', 'sadness'], 'shipwreck': ['0.562', 'sadness'], 'deceitful': ['0.562', 'sadness'], 'urn': ['0.562', 'sadness'], 'punitive': ['0.562', 'sadness'], 'injury': ['0.562', 'sadness'], 'resentment': ['0.562', 'sadness'], 'endocarditis': ['0.562', 'sadness'], 'rheumatism': ['0.562', 'sadness'], 'infliction': ['0.562', 'sadness'], 'expire': ['0.562', 'sadness'], 'tyranny': ['0.562', 'sadness'], 'anathema': ['0.562', 'sadness'], 'pauper': ['0.562', 'sadness'], 'runaway': ['0.562', 'sadness'], 'upset': ['0.562', 'sadness'], 'departed': ['0.558', 'sadness'], 'martyr': ['0.556', 'sadness'], 'smite': ['0.555', 'sadness'], 'malaria': ['0.547', 'sadness'], 'hanging': ['0.547', 'sadness'], 'chagrin': ['0.547', 'sadness'], 'malaise': ['0.547', 'sadness'], 'memorial': ['0.547', 'sadness'], 'resignation': ['0.547', 'sadness'], 'absence': ['0.547', 'sadness'], 'imissyou': ['0.547', 'sadness'], 'bummed': ['0.547', 'sadness'], 'unkind': ['0.547', 'sadness'], 'isolated': ['0.547', 'sadness'], 'plague': ['0.547', 'sadness'], 'shot': ['0.547', 'sadness'], 'bomber': ['0.547', 'sadness'], 'hydrocephalus': ['0.547', 'sadness'], 'surrendering': ['0.547', 'sadness'], 'unfulfilled': ['0.547', 'sadness'], 'discourage': ['0.547', 'sadness'], 'disillusionment': ['0.547', 'sadness'], 'reject': ['0.547', 'sadness'], 'shun': ['0.547', 'sadness'], 'pity': ['0.547', 'sadness'], 'glum': ['0.547', 'sadness'], 'nefarious': ['0.546', 'sadness'], 'groan': ['0.545', 'sadness'], 'concussion': ['0.545', 'sadness'], 'dark': ['0.545', 'sadness'], 'incrimination': ['0.545', 'sadness'], 'weakly': ['0.544', 'sadness'], 'aching': ['0.544', 'sadness'], 'discontent': ['0.543', 'sadness'], 'undertaker': ['0.538', 'sadness'], 'assailant': ['0.536', 'sadness'], 'deterioration': ['0.536', 'sadness'], 'sorely': ['0.531', 'sadness'], 'antisocial': ['0.531', 'sadness'], 'homesick': ['0.531', 'sadness'], 'confined': ['0.531', 'sadness'], 'inimical': ['0.531', 'sadness'], 'attacking': ['0.531', 'sadness'], 'tarnish': ['0.531', 'sadness'], 'forfeiture': ['0.531', 'sadness'], 'theft': ['0.531', 'sadness'], 'outburst': ['0.531', 'sadness'], 'fooled': ['0.531', 'sadness'], 'disgust': ['0.531', 'sadness'], 'embolism': ['0.531', 'sadness'], 'requiem': ['0.531', 'sadness'], 'console': ['0.531', 'sadness'], 'dispossessed': ['0.531', 'sadness'], 'disparity': ['0.531', 'sadness'], 'sick': ['0.531', 'sadness'], 'prisoner': ['0.531', 'sadness'], 'embarrassment': ['0.531', 'sadness'], 'ache': ['0.531', 'sadness'], 'inflict': ['0.531', 'sadness'], 'neurosis': ['0.531', 'sadness'], 'epitaph': ['0.531', 'sadness'], 'penance': ['0.531', 'sadness'], 'sullen': ['0.531', 'sadness'], 'grievance': ['0.530', 'sadness'], 'relapse': ['0.530', 'sadness'], 'forgotten': ['0.530', 'sadness'], 'unpleasant': ['0.530', 'sadness'], 'disable': ['0.529', 'sadness'], 'defenseless': ['0.526', 'sadness'], 'defunct': ['0.518', 'sadness'], 'ridicule': ['0.518', 'sadness'], 'misfortune': ['0.516', 'sadness'], 'blindness': ['0.516', 'sadness'], 'unfriendly': ['0.516', 'sadness'], 'delusion': ['0.516', 'sadness'], 'wither': ['0.516', 'sadness'], 'stifled': ['0.516', 'sadness'], 'elimination': ['0.516', 'sadness'], 'unlucky': ['0.516', 'sadness'], 'sore': ['0.516', 'sadness'], 'retard': ['0.516', 'sadness'], 'vegetative': ['0.516', 'sadness'], 'stripped': ['0.516', 'sadness'], 'sin': ['0.516', 'sadness'], 'sequestration': ['0.516', 'sadness'], 'displeased': ['0.516', 'sadness'], 'accident': ['0.516', 'sadness'], 'dumps': ['0.516', 'sadness'], 'hideous': ['0.516', 'sadness'], 'weakness': ['0.516', 'sadness'], 'decayed': ['0.516', 'sadness'], 'unrequited': ['0.509', 'sadness'], 'dictatorship': ['0.509', 'sadness'], 'complain': ['0.509', 'sadness'], 'lose': ['0.509', 'sadness'], 'regrettable': ['0.509', 'sadness'], 'insecure': ['0.509', 'sadness'], 'witchcraft': ['0.508', 'sadness'], 'drugged': ['0.500', 'sadness'], 'disrespectful': ['0.500', 'sadness'], 'unfair': ['0.500', 'sadness'], 'gonorrhea': ['0.500', 'sadness'], 'disturbed': ['0.500', 'sadness'], 'rot': ['0.500', 'sadness'], 'negative': ['0.500', 'sadness'], 'howl': ['0.500', 'sadness'], 'dolor': ['0.500', 'sadness'], 'mortality': ['0.500', 'sadness'], 'mad': ['0.500', 'sadness'], 'atherosclerosis': ['0.500', 'sadness'], 'impossible': ['0.500', 'sadness'], 'crash': ['0.500', 'sadness'], 'injurious': ['0.500', 'sadness'], 'chronic': ['0.500', 'sadness'], 'frowning': ['0.500', 'sadness'], 'discomfort': ['0.500', 'sadness'], 'intolerant': ['0.500', 'sadness'], 'ungodly': ['0.500', 'sadness'], 'aftermath': ['0.500', 'sadness'], 'explode': ['0.500', 'sadness'], 'cringe': ['0.500', 'sadness'], 'battled': ['0.500', 'sadness'], 'deport': ['0.500', 'sadness'], 'nauseous': ['0.500', 'sadness'], 'exclusion': ['0.500', 'sadness'], 'aggravating': ['0.500', 'sadness'], 'senile': ['0.500', 'sadness'], 'anxiety': ['0.500', 'sadness'], 'weary': ['0.500', 'sadness'], 'cytomegalovirus': ['0.500', 'sadness'], 'prosecute': ['0.500', 'sadness'], 'difficulty': ['0.500', 'sadness'], 'bier': ['0.500', 'sadness'], 'bankruptcy': ['0.500', 'sadness'], 'endemic': ['0.500', 'sadness'], 'offended': ['0.500', 'sadness'], 'damper': ['0.500', 'sadness'], 'messedup': ['0.500', 'sadness'], 'coma': ['0.500', 'sadness'], 'evict': ['0.500', 'sadness'], 'derogation': ['0.491', 'sadness'], 'rob': ['0.491', 'sadness'], 'shriek': ['0.485', 'sadness'], 'recession': ['0.485', 'sadness'], 'evasion': ['0.484', 'sadness'], 'wrongful': ['0.484', 'sadness'], 'resign': ['0.484', 'sadness'], 'coward': ['0.484', 'sadness'], 'moan': ['0.484', 'sadness'], 'weariness': ['0.484', 'sadness'], 'inadequate': ['0.484', 'sadness'], 'disturbance': ['0.484', 'sadness'], 'insult': ['0.484', 'sadness'], 'frailty': ['0.484', 'sadness'], 'adversity': ['0.484', 'sadness'], 'repress': ['0.484', 'sadness'], 'wince': ['0.484', 'sadness'], 'worn': ['0.484', 'sadness'], 'nasty': ['0.484', 'sadness'], 'sabotage': ['0.484', 'sadness'], 'criticism': ['0.484', 'sadness'], 'arsenic': ['0.484', 'sadness'], 'reprisal': ['0.484', 'sadness'], 'beg': ['0.484', 'sadness'], 'hospital': ['0.484', 'sadness'], 'offense': ['0.484', 'sadness'], 'broke': ['0.484', 'sadness'], 'infectious': ['0.483', 'sadness'], 'dishonest': ['0.482', 'sadness'], 'decay': ['0.482', 'sadness'], 'dissolution': ['0.480', 'sadness'], 'lowest': ['0.478', 'sadness'], 'unhealthy': ['0.474', 'sadness'], 'irritation': ['0.470', 'sadness'], 'perversion': ['0.469', 'sadness'], 'disapproval': ['0.469', 'sadness'], 'moody': ['0.469', 'sadness'], 'vulnerability': ['0.469', 'sadness'], 'penal': ['0.469', 'sadness'], 'domination': ['0.469', 'sadness'], 'unfavorable': ['0.469', 'sadness'], 'illegal': ['0.469', 'sadness'], 'uncaring': ['0.469', 'sadness'], 'leftout': ['0.469', 'sadness'], 'segregate': ['0.469', 'sadness'], 'collusion': ['0.469', 'sadness'], 'unfortunate': ['0.469', 'sadness'], 'sedition': ['0.469', 'sadness'], 'penalty': ['0.469', 'sadness'], 'pernicious': ['0.469', 'sadness'], 'ail': ['0.469', 'sadness'], 'conflict': ['0.469', 'sadness'], 'dashed': ['0.469', 'sadness'], 'uneasiness': ['0.469', 'sadness'], 'convict': ['0.469', 'sadness'], 'collapse': ['0.469', 'sadness'], 'fallout': ['0.469', 'sadness'], 'expulsion': ['0.469', 'sadness'], 'frustrate': ['0.469', 'sadness'], 'criticize': ['0.469', 'sadness'], 'measles': ['0.469', 'sadness'], 'recidivism': ['0.469', 'sadness'], 'frayed': ['0.467', 'sadness'], 'infamy': ['0.464', 'sadness'], 'plunder': ['0.461', 'sadness'], 'depreciated': ['0.460', 'sadness'], 'wane': ['0.456', 'sadness'], 'badly': ['0.455', 'sadness'], 'unlawful': ['0.455', 'sadness'], 'gone': ['0.453', 'sadness'], 'scarcity': ['0.453', 'sadness'], 'secluded': ['0.453', 'sadness'], 'memorials': ['0.453', 'sadness'], 'surrender': ['0.453', 'sadness'], 'inability': ['0.453', 'sadness'], 'tribulation': ['0.453', 'sadness'], 'perplexity': ['0.453', 'sadness'], 'inhospitable': ['0.453', 'sadness'], 'invade': ['0.453', 'sadness'], 'worse': ['0.453', 'sadness'], 'disapprove': ['0.453', 'sadness'], 'wrongly': ['0.453', 'sadness'], 'revolver': ['0.453', 'sadness'], 'vulgarity': ['0.453', 'sadness'], 'bittersweet': ['0.453', 'sadness'], 'discriminate': ['0.453', 'sadness'], 'foreclose': ['0.453', 'sadness'], 'upheaval': ['0.453', 'sadness'], 'wreck': ['0.453', 'sadness'], 'despotism': ['0.453', 'sadness'], 'fell': ['0.453', 'sadness'], 'frown': ['0.453', 'sadness'], 'confiscate': ['0.453', 'sadness'], 'criticise': ['0.453', 'sadness'], 'doldrums': ['0.453', 'sadness'], 'refugee': ['0.453', 'sadness'], 'avalanche': ['0.453', 'sadness'], 'lowly': ['0.448', 'sadness'], 'lone': ['0.446', 'sadness'], 'guillotine': ['0.446', 'sadness'], 'encumbrance': ['0.441', 'sadness'], 'annulment': ['0.440', 'sadness'], 'delirious': ['0.439', 'sadness'], 'confinement': ['0.439', 'sadness'], 'badday': ['0.439', 'sadness'], 'bummer': ['0.439', 'sadness'], 'underpaid': ['0.438', 'sadness'], 'detainee': ['0.438', 'sadness'], 'restriction': ['0.438', 'sadness'], 'stigma': ['0.438', 'sadness'], 'fugitive': ['0.438', 'sadness'], 'sympathize': ['0.438', 'sadness'], 'indigent': ['0.438', 'sadness'], 'bum': ['0.438', 'sadness'], 'pensive': ['0.438', 'sadness'], 'paucity': ['0.438', 'sadness'], 'blues': ['0.438', 'sadness'], 'emotional': ['0.438', 'sadness'], 'diminish': ['0.438', 'sadness'], 'disapproving': ['0.438', 'sadness'], 'disapproved': ['0.438', 'sadness'], 'darkened': ['0.438', 'sadness'], 'apathetic': ['0.438', 'sadness'], 'imprudent': ['0.438', 'sadness'], 'abscess': ['0.438', 'sadness'], 'insignificant': ['0.438', 'sadness'], 'animosity': ['0.438', 'sadness'], 'cancellation': ['0.438', 'sadness'], 'problem': ['0.438', 'sadness'], 'forbid': ['0.438', 'sadness'], 'lunacy': ['0.438', 'sadness'], 'dislocated': ['0.438', 'sadness'], 'disagreement': ['0.438', 'sadness'], 'lethargy': ['0.438', 'sadness'], 'rejects': ['0.438', 'sadness'], 'disconnected': ['0.438', 'sadness'], 'absent': ['0.438', 'sadness'], 'departure': ['0.438', 'sadness'], 'ghetto': ['0.438', 'sadness'], 'unattainable': ['0.438', 'sadness'], 'futile': ['0.438', 'sadness'], 'coercion': ['0.438', 'sadness'], 'deflate': ['0.438', 'sadness'], 'insolvency': ['0.438', 'sadness'], 'farewell': ['0.438', 'sadness'], 'appendicitis': ['0.438', 'sadness'], 'bothering': ['0.435', 'sadness'], 'disqualified': ['0.435', 'sadness'], 'tripping': ['0.429', 'sadness'], 'sunk': ['0.426', 'sadness'], 'draining': ['0.424', 'sadness'], 'lastday': ['0.424', 'sadness'], 'varicella': ['0.424', 'sadness'], 'retribution': ['0.424', 'sadness'], 'cardiomyopathy': ['0.422', 'sadness'], 'scarce': ['0.422', 'sadness'], 'thief': ['0.422', 'sadness'], 'unequal': ['0.422', 'sadness'], 'cutting': ['0.422', 'sadness'], 'neuralgia': ['0.422', 'sadness'], 'unwelcome': ['0.422', 'sadness'], 'haunted': ['0.422', 'sadness'], 'insolvent': ['0.422', 'sadness'], 'rip': ['0.422', 'sadness'], 'cyst': ['0.422', 'sadness'], 'jarring': ['0.422', 'sadness'], 'deviation': ['0.422', 'sadness'], 'wrongdoing': ['0.422', 'sadness'], 'bad': ['0.422', 'sadness'], 'handicap': ['0.422', 'sadness'], 'dispassionate': ['0.422', 'sadness'], 'falling': ['0.422', 'sadness'], 'beggar': ['0.422', 'sadness'], 'difficulties': ['0.421', 'sadness'], 'invader': ['0.420', 'sadness'], 'drab': ['0.420', 'sadness'], 'fall': ['0.418', 'sadness'], 'illegitimate': ['0.416', 'sadness'], 'expel': ['0.414', 'sadness'], 'darkness': ['0.409', 'sadness'], 'meaningless': ['0.409', 'sadness'], 'syncope': ['0.407', 'sadness'], 'obnoxious': ['0.406', 'sadness'], 'darken': ['0.406', 'sadness'], 'enmity': ['0.406', 'sadness'], 'bitch': ['0.406', 'sadness'], 'confine': ['0.406', 'sadness'], 'hoax': ['0.406', 'sadness'], 'precarious': ['0.406', 'sadness'], 'feudalism': ['0.406', 'sadness'], 'wildfire': ['0.406', 'sadness'], 'fatigued': ['0.406', 'sadness'], 'fault': ['0.406', 'sadness'], 'stroke': ['0.406', 'sadness'], 'subjected': ['0.406', 'sadness'], 'fury': ['0.406', 'sadness'], 'unsatisfied': ['0.405', 'sadness'], 'spank': ['0.403', 'sadness'], 'deluge': ['0.402', 'sadness'], 'sigh': ['0.402', 'sadness'], 'spinster': ['0.400', 'sadness'], 'blue': ['0.400', 'sadness'], 'owing': ['0.398', 'sadness'], 'needalife': ['0.398', 'sadness'], 'embarrass': ['0.394', 'sadness'], 'pitfall': ['0.394', 'sadness'], 'seriousness': ['0.394', 'sadness'], 'pointless': ['0.394', 'sadness'], 'cage': ['0.391', 'sadness'], 'brute': ['0.391', 'sadness'], 'exhausted': ['0.391', 'sadness'], 'debt': ['0.391', 'sadness'], 'condescension': ['0.391', 'sadness'], 'reproach': ['0.391', 'sadness'], 'noose': ['0.391', 'sadness'], 'insulting': ['0.391', 'sadness'], 'ifonly': ['0.391', 'sadness'], 'stretcher': ['0.391', 'sadness'], 'trickery': ['0.391', 'sadness'], 'punch': ['0.391', 'sadness'], 'coldness': ['0.391', 'sadness'], 'dwarfed': ['0.391', 'sadness'], 'ravenous': ['0.391', 'sadness'], 'feeble': ['0.391', 'sadness'], 'inefficient': ['0.391', 'sadness'], 'refused': ['0.391', 'sadness'], 'daemon': ['0.391', 'sadness'], 'banshee': ['0.391', 'sadness'], 'monsoon': ['0.391', 'sadness'], 'rue': ['0.391', 'sadness'], 'ineptitude': ['0.391', 'sadness'], 'subvert': ['0.384', 'sadness'], 'jealousy': ['0.382', 'sadness'], 'geriatric': ['0.379', 'sadness'], 'miss': ['0.379', 'sadness'], 'struggle': ['0.379', 'sadness'], 'inexcusable': ['0.379', 'sadness'], 'entangled': ['0.377', 'sadness'], 'descent': ['0.377', 'sadness'], 'ashes': ['0.377', 'sadness'], 'inconsiderate': ['0.375', 'sadness'], 'sucks': ['0.375', 'sadness'], 'blackness': ['0.375', 'sadness'], 'slump': ['0.375', 'sadness'], 'noncompliance': ['0.375', 'sadness'], 'scar': ['0.375', 'sadness'], 'murky': ['0.375', 'sadness'], 'funk': ['0.375', 'sadness'], 'landslide': ['0.375', 'sadness'], 'disqualify': ['0.375', 'sadness'], 'wasting': ['0.375', 'sadness'], 'exhaustion': ['0.375', 'sadness'], 'goodbye': ['0.375', 'sadness'], 'sympathy': ['0.375', 'sadness'], 'oust': ['0.375', 'sadness'], 'parting': ['0.375', 'sadness'], 'withdraw': ['0.375', 'sadness'], 'inferior': ['0.375', 'sadness'], 'prostitution': ['0.375', 'sadness'], 'dispel': ['0.375', 'sadness'], 'overcast': ['0.375', 'sadness'], 'rabid': ['0.375', 'sadness'], 'unattractive': ['0.373', 'sadness'], 'delay': ['0.373', 'sadness'], 'apologize': ['0.370', 'sadness'], 'crazy': ['0.368', 'sadness'], 'bastard': ['0.366', 'sadness'], 'deteriorated': ['0.365', 'sadness'], 'inter': ['0.364', 'sadness'], 'empty': ['0.364', 'sadness'], 'mocking': ['0.363', 'sadness'], 'adder': ['0.361', 'sadness'], 'perpetrator': ['0.359', 'sadness'], 'hindering': ['0.359', 'sadness'], 'affront': ['0.359', 'sadness'], 'arraignment': ['0.359', 'sadness'], 'fruitless': ['0.359', 'sadness'], 'unable': ['0.359', 'sadness'], 'disconnect': ['0.359', 'sadness'], 'defendant': ['0.359', 'sadness'], 'corse': ['0.359', 'sadness'], 'obesity': ['0.359', 'sadness'], 'taunt': ['0.359', 'sadness'], 'servile': ['0.359', 'sadness'], 'misunderstanding': ['0.359', 'sadness'], 'austere': ['0.359', 'sadness'], 'doubt': ['0.359', 'sadness'], 'wrangling': ['0.359', 'sadness'], 'hunter': ['0.359', 'sadness'], 'unsuccessful': ['0.359', 'sadness'], 'inefficiency': ['0.359', 'sadness'], 'consecration': ['0.359', 'sadness'], 'tremor': ['0.359', 'sadness'], 'unemployed': ['0.359', 'sadness'], 'fuss': ['0.359', 'sadness'], 'unpopular': ['0.359', 'sadness'], 'fainting': ['0.359', 'sadness'], 'numbness': ['0.359', 'sadness'], 'flounder': ['0.359', 'sadness'], 'idiocy': ['0.359', 'sadness'], 'lockup': ['0.359', 'sadness'], 'plaintive': ['0.359', 'sadness'], 'unrest': ['0.351', 'sadness'], 'spoiler': ['0.348', 'sadness'], 'intervention': ['0.348', 'sadness'], 'waste': ['0.348', 'sadness'], 'wimpy': ['0.348', 'sadness'], 'absentee': ['0.348', 'sadness'], 'flaw': ['0.347', 'sadness'], 'desert': ['0.344', 'sadness'], 'cumbersome': ['0.344', 'sadness'], 'specter': ['0.344', 'sadness'], 'resigned': ['0.344', 'sadness'], 'furrow': ['0.344', 'sadness'], 'lagging': ['0.344', 'sadness'], 'forfeit': ['0.344', 'sadness'], 'uninspired': ['0.344', 'sadness'], 'plea': ['0.344', 'sadness'], 'intercede': ['0.344', 'sadness'], 'stained': ['0.344', 'sadness'], 'litigate': ['0.344', 'sadness'], 'blindly': ['0.344', 'sadness'], 'attenuation': ['0.344', 'sadness'], 'militia': ['0.344', 'sadness'], 'surgery': ['0.344', 'sadness'], 'detention': ['0.344', 'sadness'], 'lawsuit': ['0.344', 'sadness'], 'thrash': ['0.344', 'sadness'], 'uninvited': ['0.344', 'sadness'], 'unaccountable': ['0.339', 'sadness'], 'myopia': ['0.339', 'sadness'], 'mishap': ['0.338', 'sadness'], 'probation': ['0.336', 'sadness'], 'severance': ['0.333', 'sadness'], 'disagreeing': ['0.333', 'sadness'], 'incompetent': ['0.328', 'sadness'], 'nether': ['0.328', 'sadness'], 'endless': ['0.328', 'sadness'], 'dependence': ['0.328', 'sadness'], 'disallowed': ['0.328', 'sadness'], 'bondage': ['0.328', 'sadness'], 'soreness': ['0.328', 'sadness'], 'unacknowledged': ['0.328', 'sadness'], 'squall': ['0.328', 'sadness'], 'unacceptable': ['0.328', 'sadness'], 'adrift': ['0.328', 'sadness'], 'nepotism': ['0.328', 'sadness'], 'sterile': ['0.328', 'sadness'], 'bacteria': ['0.328', 'sadness'], 'leave': ['0.328', 'sadness'], 'scold': ['0.327', 'sadness'], 'flaccid': ['0.324', 'sadness'], 'hobo': ['0.323', 'sadness'], 'fragile': ['0.319', 'sadness'], 'stingy': ['0.319', 'sadness'], 'sue': ['0.318', 'sadness'], 'scarcely': ['0.318', 'sadness'], 'wan': ['0.312', 'sadness'], 'committal': ['0.312', 'sadness'], 'mistake': ['0.312', 'sadness'], 'clouded': ['0.312', 'sadness'], 'skid': ['0.312', 'sadness'], 'defy': ['0.312', 'sadness'], 'thresh': ['0.312', 'sadness'], 'fatty': ['0.312', 'sadness'], 'nostalgia': ['0.312', 'sadness'], 'inhibit': ['0.312', 'sadness'], 'evanescence': ['0.312', 'sadness'], 'ulcer': ['0.312', 'sadness'], 'hamstring': ['0.312', 'sadness'], 'nonsensical': ['0.312', 'sadness'], 'conceal': ['0.311', 'sadness'], 'blemish': ['0.310', 'sadness'], 'resisting': ['0.309', 'sadness'], 'sympathetic': ['0.307', 'sadness'], 'bugaboo': ['0.304', 'sadness'], 'confess': ['0.303', 'sadness'], 'opium': ['0.303', 'sadness'], 'alas': ['0.302', 'sadness'], 'incase': ['0.297', 'sadness'], 'halting': ['0.297', 'sadness'], 'incompatible': ['0.297', 'sadness'], 'migraine': ['0.297', 'sadness'], 'mislead': ['0.297', 'sadness'], 'toocold': ['0.297', 'sadness'], 'suppress': ['0.297', 'sadness'], 'inappropriate': ['0.297', 'sadness'], 'discontinuity': ['0.297', 'sadness'], 'setback': ['0.297', 'sadness'], 'dull': ['0.297', 'sadness'], 'weak': ['0.297', 'sadness'], 'subsidence': ['0.297', 'sadness'], 'wrinkled': ['0.297', 'sadness'], 'hermit': ['0.296', 'sadness'], 'moving': ['0.295', 'sadness'], 'shrink': ['0.295', 'sadness'], 'shiver': ['0.291', 'sadness'], 'tramp': ['0.288', 'sadness'], 'unimportant': ['0.288', 'sadness'], 'constraint': ['0.288', 'sadness'], 'rubble': ['0.282', 'sadness'], 'negro': ['0.281', 'sadness'], 'grey': ['0.281', 'sadness'], 'flinch': ['0.281', 'sadness'], 'apathy': ['0.281', 'sadness'], 'confession': ['0.281', 'sadness'], 'down': ['0.281', 'sadness'], 'remove': ['0.281', 'sadness'], 'unseat': ['0.281', 'sadness'], 'wearily': ['0.281', 'sadness'], 'taint': ['0.281', 'sadness'], 'excluding': ['0.281', 'sadness'], 'overdue': ['0.281', 'sadness'], 'shortage': ['0.281', 'sadness'], 'grumpy': ['0.281', 'sadness'], 'flop': ['0.281', 'sadness'], 'revoke': ['0.281', 'sadness'], 'adverse': ['0.281', 'sadness'], 'black': ['0.281', 'sadness'], 'scrapie': ['0.281', 'sadness'], 'timid': ['0.281', 'sadness'], 'senseless': ['0.281', 'sadness'], 'knell': ['0.275', 'sadness'], 'soldier': ['0.273', 'sadness'], 'humbled': ['0.273', 'sadness'], 'confusion': ['0.273', 'sadness'], 'throb': ['0.273', 'sadness'], 'jurisprudence': ['0.273', 'sadness'], 'gray': ['0.269', 'sadness'], 'shack': ['0.266', 'sadness'], 'mixedemotions': ['0.266', 'sadness'], 'obstacle': ['0.266', 'sadness'], 'lax': ['0.266', 'sadness'], 'remiss': ['0.266', 'sadness'], 'slur': ['0.266', 'sadness'], 'unrealistic': ['0.266', 'sadness'], 'drifted': ['0.266', 'sadness'], 'eternity': ['0.266', 'sadness'], 'leaving': ['0.266', 'sadness'], 'inconvenient': ['0.263', 'sadness'], 'misrepresentation': ['0.259', 'sadness'], 'restrict': ['0.259', 'sadness'], 'stagnant': ['0.259', 'sadness'], 'disservice': ['0.258', 'sadness'], 'nosun': ['0.255', 'sadness'], 'backwater': ['0.255', 'sadness'], 'wilderness': ['0.255', 'sadness'], 'error': ['0.250', 'sadness'], 'anchorage': ['0.250', 'sadness'], 'unexplained': ['0.250', 'sadness'], 'humbug': ['0.250', 'sadness'], 'gullible': ['0.250', 'sadness'], 'speculation': ['0.250', 'sadness'], 'communism': ['0.250', 'sadness'], 'uneducated': ['0.250', 'sadness'], 'tempest': ['0.250', 'sadness'], 'bang': ['0.250', 'sadness'], 'labored': ['0.250', 'sadness'], 'incomplete': ['0.250', 'sadness'], 'wasteful': ['0.250', 'sadness'], 'pine': ['0.250', 'sadness'], 'undying': ['0.250', 'sadness'], 'older': ['0.250', 'sadness'], 'demonstrative': ['0.242', 'sadness'], 'melodrama': ['0.242', 'sadness'], 'rainyday': ['0.242', 'sadness'], 'necessity': ['0.236', 'sadness'], 'boredom': ['0.235', 'sadness'], 'cloudy': ['0.234', 'sadness'], 'hollow': ['0.234', 'sadness'], 'burke': ['0.234', 'sadness'], 'trash': ['0.234', 'sadness'], 'pale': ['0.234', 'sadness'], 'depart': ['0.234', 'sadness'], 'uninteresting': ['0.234', 'sadness'], 'sentence': ['0.234', 'sadness'], 'void': ['0.234', 'sadness'], 'cancel': ['0.234', 'sadness'], 'foggy': ['0.234', 'sadness'], 'warp': ['0.234', 'sadness'], 'misty': ['0.234', 'sadness'], 'blockade': ['0.234', 'sadness'], 'healing': ['0.234', 'sadness'], 'case': ['0.228', 'sadness'], 'rainy': ['0.227', 'sadness'], 'onerous': ['0.223', 'sadness'], 'bottom': ['0.223', 'sadness'], 'uninterested': ['0.223', 'sadness'], 'fasting': ['0.220', 'sadness'], 'coping': ['0.219', 'sadness'], 'discolored': ['0.219', 'sadness'], 'thirst': ['0.219', 'sadness'], 'boooo': ['0.219', 'sadness'], 'pious': ['0.219', 'sadness'], 'blunder': ['0.219', 'sadness'], 'indifference': ['0.219', 'sadness'], 'dole': ['0.219', 'sadness'], 'cocaine': ['0.218', 'sadness'], 'tough': ['0.212', 'sadness'], 'revolution': ['0.203', 'sadness'], 'fat': ['0.203', 'sadness'], 'arid': ['0.203', 'sadness'], 'sluggish': ['0.203', 'sadness'], 'yucky': ['0.203', 'sadness'], 'sprain': ['0.203', 'sadness'], 'chilly': ['0.203', 'sadness'], 'lower': ['0.203', 'sadness'], 'chargeable': ['0.203', 'sadness'], 'hoary': ['0.203', 'sadness'], 'wanting': ['0.202', 'sadness'], 'progression': ['0.201', 'sadness'], 'closure': ['0.195', 'sadness'], 'unbeaten': ['0.193', 'sadness'], 'rack': ['0.188', 'sadness'], 'halter': ['0.188', 'sadness'], 'meh': ['0.188', 'sadness'], 'cold': ['0.188', 'sadness'], 'tease': ['0.188', 'sadness'], 'splitting': ['0.188', 'sadness'], 'rumor': ['0.188', 'sadness'], 'cataract': ['0.188', 'sadness'], 'invalid': ['0.188', 'sadness'], 'heartfelt': ['0.188', 'sadness'], 'oddity': ['0.188', 'sadness'], 'veal': ['0.188', 'sadness'], 'retirement': ['0.188', 'sadness'], 'interrupted': ['0.188', 'sadness'], 'concerned': ['0.184', 'sadness'], 'sarcasm': ['0.181', 'sadness'], 'strip': ['0.179', 'sadness'], 'feeling': ['0.172', 'sadness'], 'sap': ['0.172', 'sadness'], 'memories': ['0.172', 'sadness'], 'eschew': ['0.172', 'sadness'], 'esteem': ['0.172', 'sadness'], 'cupping': ['0.172', 'sadness'], 'overload': ['0.172', 'sadness'], 'divided': ['0.172', 'sadness'], 'destination': ['0.170', 'sadness'], 'nosnow': ['0.169', 'sadness'], 'limited': ['0.167', 'sadness'], 'rain': ['0.163', 'sadness'], 'willful': ['0.160', 'sadness'], 'untitled': ['0.157', 'sadness'], 'stint': ['0.156', 'sadness'], 'weeds': ['0.156', 'sadness'], 'cross': ['0.156', 'sadness'], 'pare': ['0.155', 'sadness'], 'snort': ['0.154', 'sadness'], 'procession': ['0.152', 'sadness'], 'inconsequential': ['0.152', 'sadness'], 'tax': ['0.142', 'sadness'], 'overpriced': ['0.141', 'sadness'], 'lesbian': ['0.141', 'sadness'], 'weight': ['0.141', 'sadness'], 'tolerate': ['0.141', 'sadness'], 'mug': ['0.141', 'sadness'], 'emo': ['0.141', 'sadness'], 'touchy': ['0.140', 'sadness'], 'grounded': ['0.130', 'sadness'], 'kennel': ['0.130', 'sadness'], 'commemorate': ['0.125', 'sadness'], 'late': ['0.125', 'sadness'], 'theocratic': ['0.125', 'sadness'], 'margin': ['0.125', 'sadness'], 'socialist': ['0.125', 'sadness'], 'stillness': ['0.125', 'sadness'], 'meek': ['0.125', 'sadness'], 'terrific': ['0.125', 'sadness'], 'sisterhood': ['0.125', 'sadness'], 'clouds': ['0.125', 'sadness'], 'unpaid': ['0.125', 'sadness'], 'default': ['0.121', 'sadness'], 'lace': ['0.118', 'sadness'], 'unpublished': ['0.116', 'sadness'], 'interested': ['0.114', 'sadness'], 'fortress': ['0.110', 'sadness'], 'fleece': ['0.109', 'sadness'], 'priesthood': ['0.109', 'sadness'], 'rating': ['0.109', 'sadness'], 'ultimate': ['0.109', 'sadness'], 'lush': ['0.109', 'sadness'], 'orchestra': ['0.109', 'sadness'], 'harry': ['0.109', 'sadness'], 'sanctify': ['0.108', 'sadness'], 'income': ['0.100', 'sadness'], 'winning': ['0.094', 'sadness'], 'quiet': ['0.094', 'sadness'], 'sonnet': ['0.094', 'sadness'], 'boo': ['0.094', 'sadness'], 'vainly': ['0.091', 'sadness'], 'hut': ['0.078', 'sadness'], 'opera': ['0.078', 'sadness'], 'humble': ['0.078', 'sadness'], 'motivating': ['0.078', 'sadness'], 'wet': ['0.078', 'sadness'], 'ovation': ['0.078', 'sadness'], 'hug': ['0.078', 'sadness'], 'treat': ['0.076', 'sadness'], 'hymn': ['0.064', 'sadness'], 'honest': ['0.062', 'sadness'], 'relics': ['0.061', 'sadness'], 'couch': ['0.060', 'sadness'], 'waffle': ['0.047', 'sadness'], 'shell': ['0.045', 'sadness'], 'musical': ['0.045', 'sadness'], 'savor': ['0.034', 'sadness'], 'napkin': ['0.031', 'sadness'], 'vote': ['0.031', 'sadness'], 'sing': ['0.017', 'sadness'], 'music': ['0.016', 'sadness'], 'mother': ['0.016', 'sadness'], 'nutritious': ['0.015', 'sadness'], 'lovely': ['0.009', 'sadness'], 'liquor': ['0.000', 'sadness'], 'sweetheart': ['0.000', 'sadness'], 'romance': ['0.000', 'sadness'], 'art': ['0.000', 'sadness']}
#
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
emotion_model = tf.keras.models.load_model('late_fusion2_model.h5')

#
WORD = re.compile(r"\w+")

model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)

token_path = "Flickr8k.token.txt"
train_images_path = 'Flickr_8k.trainImages.txt'
doc = open(token_path, 'r').read()
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
        desc_list[i] = ' '.join(desc)

vocabulary = set()
for key in descriptions.keys():
    [vocabulary.update(d.split()) for d in descriptions[key]]

lines = list()
for key, desc_list in descriptions.items():
    for desc in desc_list:
        lines.append(key + ' ' + desc)
new_descriptions = '\n'.join(lines)

doc = open(train_images_path, 'r').read()
dataset = list()
for line in doc.split('\n'):
    if len(line) > 1:
        identifier = line.split('.')[0]
        dataset.append(identifier)

train = set(dataset)

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


def send_image_for_reshape(img):
    encoding = {}
    encoding[0] = encode(img)

 #   p = list(encoding.keys())
 #   pic = p[0]
    image_for_caption = encoding[0].reshape((1, 2048))

    return image_for_caption


def preprocess(image_path):
    #  img = image.load_img(image_path, target_size=(299, 299))
    size = (299,299)
    img = ImageOps.fit(image_path, size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def encode(image):
    image = preprocess(image)
    fea_vec = model_new.predict(image)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec


c_model = load_model('model_15.h5')


def beam_search_predictions(image, beam_index=3):
    start = [wordtoix["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = c_model.predict([image, par_caps], verbose=0)
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


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def cosine_similarity(text1,text2):

    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)

    cosine = get_cosine(vector1, vector2)
    print("Cosine similarity value :", cosine)
    if cosine>0.1:
        return True
    else:
        return False


def image_emotion_prediction(Image_from_tweet):
    if Image_from_tweet is not None:
        ## Image preprocessing
        image = Image_from_tweet
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
        df = pd.DataFrame({'emotion_classes': labels, 'predicted_probability': preds[0]})
        k = df['predicted_probability'].values.tolist()
        s = pd.Series(k)
        m = (s * 100).tolist()
        return predicted_classes, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]


def tweet_preprocessing(tweet):
    tweet = re.sub("@[a-z0-9A-Z]+", "", tweet)
    tweet = tweet.replace("\\n", " ")
    tweet = re.sub("#", "", tweet)
    tweet = re.sub("[^a-z A-Z ']", "", tweet)
    tweet = re.sub("[\s]+", " ", tweet)
    tweet = re.sub(r"http\S+", "", tweet)  # removal of URLs
    # removal of URLs
    tweet = re.sub(r"won't", "will not", tweet)
    tweet = re.sub(r"can\'t", "can not", tweet)
    # general
    tweet = re.sub(r"n\'t", " not", tweet)
    tweet = re.sub(r"\'re", " are", tweet)
    tweet = re.sub(r"\'s", " is", tweet)
    tweet = re.sub(r"\'d", " would", tweet)
    tweet = re.sub(r"\'ll", " will", tweet)
    tweet = re.sub(r"\'t", " not", tweet)
    tweet = re.sub(r"\'ve", " have", tweet)
    tweet = re.sub(r"\'m", " am", tweet)
    tweet = tweet.lower()
    tweet = check(tweet)
    return tweet

#negation handling

def negation_modifier(sent):
    gov = []
    depp = []
    dependents = []
    super1 = []
    flag = 0
    f = 0
    sent_splits = sent.split(" ")
    if sent_splits[0] in negators:
        if sent_splits[1] in prons:
            sent_splits[0] = "none"
            sent_splits.remove(sent_splits[1])
    hello = ' '.join(sent_splits)
    for i in sent_splits:
        f = f + 1
        if i in negators:
            if i == "not" and (sent.split(" ")[f] not in ["only", "even"]):
                flag = 1
                gov = []
                depp = []
                dependents = []
            if i in negators and (sent.split(" ")[f] == "false"):
                sent_splits[f - 1] = "true"
                sent_splits.remove("false")
                hello = ' '.join(sent_splits)
            elif i != "not":
                flag = 1
    parse, = dep_parser.raw_parse(hello)
    print("The typed dependency triples:")
    print()
    for governor, dep, dependent in parse.triples():
        print(governor[0], dep, dependent[0])
    print()
    print("The negated words are/is:")
    print()
    xxx = 0
    yyy = 1
    njj = 1
    fll = 0
    if flag == 1:
        gov = []
        depp = []
        dependents = []
        for governor, dep, dependent in parse.triples():
            # print(governor[0], dep, dependent[0])
            gov.append(governor[0])
            depp.append(dep)
            dependents.append(dependent[0])
        dependents.append("dummy")
        pp = 0
        for q in depp:
            pp = pp + 1
            if q == "det":
                if dependents[pp - 1] in negators:
                    print(gov[pp - 1])
                    # print("dets")
                    super1.append(gov[pp - 1])
                    fll = 1

            if q == "nsubj":
                if dependents[pp - 1] in negators:
                    nsubs = 0
                    for ns in depp:

                        nsubs = nsubs + 1
                        if (ns == "advmod") and dependents[nsubs - 1] in negators:

                            if dependents[nsubs - 1] in negators:

                                if (gov[nsubs - 1] == gov[pp - 1]):
                                    njj = 0
                                    fll = 1
                            if (njj == 1):
                                print(gov[pp - 1])
                                super1.append(gov[pp - 1])
                                fll = 1

                    if (njj == 1) and (fll == 0):
                        print(gov[pp - 1])
                        fll = 1
                        super1.append(gov[pp - 1])

            if q == "nsubj:pass":
                if dependents[pp - 1] in negators:
                    nsubs = 0
                    for ns in depp:
                        nsubs = nsubs + 1
                        if (ns == "advmod") and dependents[nsubs - 1] in negators:

                            if dependents[nsubs - 1] in negators:
                                if (gov[nsubs - 1] == gov[pp - 1]):
                                    njj = 0
                                    fll = 1
                            if (njj == 1):
                                print(gov[pp - 1])
                                super1.append(gov[pp - 1])
                                fll = 1
                    if (njj == 1) and (fll == 0):
                        print(gov[pp - 1])
                        fll = 1
                        super1.append(gov[pp - 1])
        m = 0
        for j in depp:
            m = m + 1
            if j == "advmod":
                if dependents[m - 1] in negators:
                    n = 0
                    bb = 0
                    if fll == 0:
                        for aa in depp:
                            bb = bb + 1
                            #ccomp-causal complement eg:He says that you like to swim.
                            if aa == "xcomp":
                                nums = 0
                                if gov[bb - 1] == gov[m - 1]:
                                    num = 0
                                    for deps in depp:
                                        num = num + 1
                                        if deps == "advmod":
                                            if (dependents[bb - 1] == gov[num - 1]):
                                                nums = 1
                                                xxx = 1
                                                fll = 1

                                    if (nums == 0):
                                        print(dependents[bb - 1])
                                        super1.append(dependents[bb - 1])
                                        fll = 1

                                if gov[bb - 1] == gov[m - 1] and xxx == 0:
                                    print(dependents[bb - 1])
                                    super1.append(dependents[bb - 1])
                                    fll = 1
                            if aa == "ccomp":
                                nums = 0
                                if gov[bb - 1] == gov[m - 1]:
                                    num = 0
                                    for deps in depp:
                                        num = num + 1
                                        if deps == "advmod":

                                            if (dependents[bb - 1] == gov[num - 1]):
                                                nums = 1
                                                fll = 1

                                    if (nums == 0):
                                        n = 0
                                        for k in depp:
                                            n = n + 1
                                            if k == "conj":
                                                o = 0
                                                for a in depp:
                                                    o = o + 1
                                                    if a == "cc":
                                                        if dependents[o - 1] in pos_conj:
                                                            if gov[o - 1] == dependents[n - 1]:
                                                                if gov[n - 1] == dependents[bb - 1]:
                                                                    super1.append(dependents[bb - 1])
                                                                    super1.append(gov[o - 1])
                                                                    print(gov[o - 1])
                                                                    print(dependents[bb - 1])
                                                                    yyy = 0

                                                    fll = 1

                                        if (yyy == 1):
                                            print(dependents[bb - 1])
                                            super1.append(dependents[bb - 1])
                                            fll = 1

                    if fll == 0:
                        n = 0
                        for k in depp:
                            n = n + 1
                            if k == "conj":
                                o = 0
                                for a in depp:
                                    o = o + 1
                                    if a == "cc":
                                        if dependents[o - 1] in pos_conj:
                                            if gov[o - 1] == dependents[n - 1]:
                                                if gov[n - 1] == gov[m - 1]:
                                                    print(gov[m - 1])
                                                    print(dependents[n - 1])
                                                    super1.append(gov[m - 1])
                                                    super1.append(dependents[n - 1])
                                                    fll = 1

                    if fll == 0:
                        print(gov[m - 1])
                        super1.append(gov[m - 1])
    return super1, hello


def scoring_tweets(tweet, emoji_score_vector):
    negated_words, input_tweet = negation_modifier(tweet)
    sentence = input_tweet
    print(sentence)
    sentence_array = sentence.split(" ")
    score_vector = np.zeros((len(sentence_array), 4))
    print(sentence_array)
    print("Scoring")
    index = 0
    m = 0
    for j in sentence_array:
        index = index + 1
        emotion = NRCLex(j)
        if "sadness" not in emotion.affect_list:
            if "joy" not in emotion.affect_list:
                if "anger" not in emotion.affect_list:
                    if "fear" not in emotion.affect_list:
                        continue
        # print(j+":")
        i = j
        if j in negated_words:
            m = 1
        if (m == 0):
            if i in diction.keys():
                score_vector[index - 1][0] = diction[i][0]
                m = 0
            else:
                for k in diction.keys():
                    lemmat_v_i = Lemmatizer.lemmatize(i, pos="v")

                    lemmatized_k = Lemmatizer.lemmatize(k, pos="v")
                    if lemmat_v_i == lemmatized_k:
                        score_vector[index - 1][0] = diction[k][0]
                        m = 0
                        break
                    lemmat_v_i = Lemmatizer.lemmatize(i, pos="n")

                    lemmatized_k = Lemmatizer.lemmatize(k, pos="n")
                    if lemmat_v_i == lemmatized_k:
                        score_vector[index - 1][0] = diction[k][0]
                        m = 0
                        break
                    lemmat_v_i = Lemmatizer.lemmatize(i, pos="r")

                    lemmatized_k = Lemmatizer.lemmatize(k, pos="r")
                    if lemmat_v_i == lemmatized_k:
                        score_vector[index - 1][0] = diction[k][0]
                        m = 0
                        break
                    lemmat_v_i = Lemmatizer.lemmatize(i, pos="a")

                    lemmatized_k = Lemmatizer.lemmatize(k, pos="a")
                    if lemmat_v_i == lemmatized_k:
                        score_vector[index - 1][0] = diction[k][0]
                        m = 0
                        break
            if i in dic.keys():
                score_vector[index - 1][1] = dic[i][0]
                m = 0
            else:
                for k in dic.keys():
                    lemmat_v_i = Lemmatizer.lemmatize(i, pos="v")

                    lemmatized_k = Lemmatizer.lemmatize(k, pos="v")
                    if lemmat_v_i == lemmatized_k:
                        score_vector[index - 1][1] = dic[k][0]
                        m = 0
                        break
                    lemmat_v_i = Lemmatizer.lemmatize(i, pos="n")

                    lemmatized_k = Lemmatizer.lemmatize(k, pos="n")
                    if lemmat_v_i == lemmatized_k:
                        score_vector[index - 1][1] = dic[k][0]
                        m = 0
                        break
                    lemmat_v_i = Lemmatizer.lemmatize(i, pos="r")

                    lemmatized_k = Lemmatizer.lemmatize(k, pos="r")
                    if lemmat_v_i == lemmatized_k:
                        score_vector[index - 1][1] = dic[k][0]
                        m = 0
                        break
                    lemmat_v_i = Lemmatizer.lemmatize(i, pos="a")

                    lemmatized_k = Lemmatizer.lemmatize(k, pos="a")
                    if lemmat_v_i == lemmatized_k:
                        score_vector[index - 1][1] = dic[k][0]
                        m = 0
                        break

            if i in di.keys():
                score_vector[index - 1][2] = di[i][0]
                m = 0
            else:
                for k in di.keys():
                    lemmat_v_i = Lemmatizer.lemmatize(i, pos="v")

                    lemmatized_k = Lemmatizer.lemmatize(k, pos="v")
                    if lemmat_v_i == lemmatized_k:
                        score_vector[index - 1][2] = di[k][0]
                        m = 0
                        break
                    lemmat_v_i = Lemmatizer.lemmatize(i, pos="n")

                    lemmatized_k = Lemmatizer.lemmatize(k, pos="n")
                    if lemmat_v_i == lemmatized_k:
                        score_vector[index - 1][2] = di[k][0]
                        # print(di[k][1])
                        m = 0
                        break
                    lemmat_v_i = Lemmatizer.lemmatize(i, pos="r")

                    lemmatized_k = Lemmatizer.lemmatize(k, pos="r")
                    if lemmat_v_i == lemmatized_k:
                        score_vector[index - 1][2] = di[k][0]
                        m = 0
                        break
                    lemmat_v_i = Lemmatizer.lemmatize(i, pos="a")

                    lemmatized_k = Lemmatizer.lemmatize(k, pos="a")
                    if lemmat_v_i == lemmatized_k:
                        score_vector[index - 1][2] = di[k][0]
                        m = 0
                        break

            if i in d.keys():
                score_vector[index - 1][3] = d[i][0]
                m = 0
            else:
                for k in d.keys():
                    lemmat_v_i = Lemmatizer.lemmatize(i, pos="v")

                    lemmatized_k = Lemmatizer.lemmatize(k, pos="v")
                    if lemmat_v_i == lemmatized_k:
                        score_vector[index - 1][3] = d[k][0]
                        m = 0
                        break
                    lemmat_v_i = Lemmatizer.lemmatize(i, pos="n")

                    lemmatized_k = Lemmatizer.lemmatize(k, pos="n")
                    if lemmat_v_i == lemmatized_k:
                        score_vector[index - 1][3] = d[k][0]
                        m = 0
                        break
                    lemmat_v_i = Lemmatizer.lemmatize(i, pos="r")

                    lemmatized_k = Lemmatizer.lemmatize(k, pos="r")
                    if lemmat_v_i == lemmatized_k:
                        score_vector[index - 1][3] = d[k][0]
                        m = 0
                        break
                    lemmat_v_i = Lemmatizer.lemmatize(i, pos="a")

                    lemmatized_k = Lemmatizer.lemmatize(k, pos="a")
                    if lemmat_v_i == lemmatized_k:
                        score_vector[index - 1][3] = d[k][0]
                        m = 0
                        break
        if (m == 1):
            if i in diction.keys():
                score_vector[index - 1][0] = (1 - float(diction[i][0]))
                m = 0
            for k in diction.keys():
                lemmat_v_i = Lemmatizer.lemmatize(i, pos="v")

                lemmatized_k = Lemmatizer.lemmatize(k, pos="v")

                if lemmat_v_i == lemmatized_k:
                    score_vector[index - 1][0] = (1 - float(diction[k][0]))
                    m = 0
                    break
                lemmat_v_i = Lemmatizer.lemmatize(i, pos="n")

                lemmatized_k = Lemmatizer.lemmatize(k, pos="n")
                if lemmat_v_i == lemmatized_k:
                    score_vector[index - 1][0] = (1 - float(diction[k][0]))
                    m = 0
                    break
                lemmat_v_i = Lemmatizer.lemmatize(i, pos="r")

                lemmatized_k = Lemmatizer.lemmatize(k, pos="r")
                if lemmat_v_i == lemmatized_k:
                    score_vector[index - 1][0] = (1 - float(diction[k][0]))
                    m = 0
                    break
                lemmat_v_i = Lemmatizer.lemmatize(i, pos="a")

                lemmatized_k = Lemmatizer.lemmatize(k, pos="a")
                if lemmat_v_i == lemmatized_k:
                    score_vector[index - 1][0] = (1 - float(diction[k][0]))
                    m = 0
                    break

            if i in dic.keys():
                score_vector[index - 1][1] = (1 - float(dic[i][0]))
                m = 0
            for k in dic.keys():
                lemmat_v_i = Lemmatizer.lemmatize(i, pos="v")

                lemmatized_k = Lemmatizer.lemmatize(k, pos="v")
                if lemmat_v_i == lemmatized_k:
                    score_vector[index - 1][1] = (1 - float(dic[k][0]))
                    m = 0
                    break
                lemmat_v_i = Lemmatizer.lemmatize(i, pos="n")

                lemmatized_k = Lemmatizer.lemmatize(k, pos="n")
                if lemmat_v_i == lemmatized_k:
                    score_vector[index - 1][1] = (1 - float(dic[k][0]))
                    m = 0
                    break
                lemmat_v_i = Lemmatizer.lemmatize(i, pos="r")

                lemmatized_k = Lemmatizer.lemmatize(k, pos="r")
                if lemmat_v_i == lemmatized_k:
                    score_vector[index - 1][1] = (1 - float(dic[k][0]))
                    m = 0
                    break
                lemmat_v_i = Lemmatizer.lemmatize(i, pos="a")

                lemmatized_k = Lemmatizer.lemmatize(k, pos="a")
                if lemmat_v_i == lemmatized_k:
                    score_vector[index - 1][1] = (1 - float(dic[k][0]))
                    m = 0
                    break

            if i in di.keys():
                score_vector[index - 1][2] = (1 - float(di[i][0]))
                m = 0

            for k in di.keys():
                lemmat_v_i = Lemmatizer.lemmatize(i, pos="v")

                lemmatized_k = Lemmatizer.lemmatize(k, pos="v")
                if lemmat_v_i == lemmatized_k:
                    score_vector[index - 1][2] = (1 - float(di[k][0]))
                    m = 0
                    break
                lemmat_v_i = Lemmatizer.lemmatize(i, pos="n")

                lemmatized_k = Lemmatizer.lemmatize(k, pos="n")
                if lemmat_v_i == lemmatized_k:
                    score_vector[index - 1][2] = (1 - float(di[k][0]))
                    m = 0
                    break
                lemmat_v_i = Lemmatizer.lemmatize(i, pos="r")

                lemmatized_k = Lemmatizer.lemmatize(k, pos="r")
                if lemmat_v_i == lemmatized_k:
                    score_vector[index - 1][2] = (1 - float(di[k][0]))
                    m = 0
                    break
                lemmat_v_i = Lemmatizer.lemmatize(i, pos="a")

                lemmatized_k = Lemmatizer.lemmatize(k, pos="a")
                if lemmat_v_i == lemmatized_k:
                    score_vector[index - 1][2] = (1 - float(di[k][0]))
                    m = 0
                    break

            if i in d.keys():
                score_vector[index - 1][3] = (1 - float(d[i][0]))
                m = 0
            for k in d.keys():
                lemmat_v_i = Lemmatizer.lemmatize(i, pos="v")

                lemmatized_k = Lemmatizer.lemmatize(k, pos="v")
                if lemmat_v_i == lemmatized_k:
                    score_vector[index - 1][3] = (1 - float(d[k][0]))
                    m = 0
                    break
                lemmat_v_i = Lemmatizer.lemmatize(i, pos="n")

                lemmatized_k = Lemmatizer.lemmatize(k, pos="n")
                if lemmat_v_i == lemmatized_k:
                    score_vector[index - 1][3] = (1 - float(d[k][0]))
                    m = 0
                    break
                lemmat_v_i = Lemmatizer.lemmatize(i, pos="r")

                lemmatized_k = Lemmatizer.lemmatize(k, pos="r")
                if lemmat_v_i == lemmatized_k:
                    score_vector[index - 1][3] = (1 - float(d[k][0]))
                    m = 0
                    break
                lemmat_v_i = Lemmatizer.lemmatize(i, pos="a")

                lemmatized_k = Lemmatizer.lemmatize(k, pos="a")
                if lemmat_v_i == lemmatized_k:
                    score_vector[index - 1][3] = (1 - float(d[k][0]))
                    m = 0
                    break

        m = 0

    print("  sad" + "   happy" + "   fear" + "   anger")
    print(score_vector)
    score_vector = np.concatenate((score_vector, emoji_score_vector))
    print("combined score")
    print(score_vector)
    s_score = 0
    h_score = 0
    f_score = 0
    a_score = 0
    s_nos = 0
    h_nos = 0
    f_nos = 0
    a_nos = 0
    for i in range(len(sentence_array)):
        if (score_vector[i][0] != 0.0):
            s_score = s_score + score_vector[i][0]
            s_nos = s_nos + 1
        if (score_vector[i][1] != 0.0):
            h_score = h_score + score_vector[i][1]
            h_nos = h_nos + 1
        if (score_vector[i][2] != 0.0):
            f_score = f_score + score_vector[i][2]
            f_nos = f_nos + 1
        if (score_vector[i][3] != 0.0):
            a_score = a_score + score_vector[i][3]
            a_nos = a_nos + 1

    if (s_nos != 0):
        s_score = s_score / s_nos
    if (h_nos != 0):
        h_score = h_score / h_nos
    if (f_nos != 0):
        f_score = f_score / f_nos
    if (a_nos != 0):
        a_score = a_score / a_nos
    hscore = h_score
    if h_score == 0:
        hscore = 0.01
    ascore = a_score
    if a_score == 0:
        ascore = 0.01
    sscore = s_score
    if s_score == 0:
        sscore = 0.01
    fscore = f_score
    if f_score == 0:
        fscore = 0.01
    return sscore, fscore, hscore, ascore, sentence_array

def fuzzification(sscore, fscore, hscore, ascore):
    h_level_lo = fuzz.interp_membership(x_h, h_lo, hscore)
    h_level_md = fuzz.interp_membership(x_h, h_md, hscore)
    h_level_hi = fuzz.interp_membership(x_h, h_hi, hscore)

    a_level_lo = fuzz.interp_membership(x_a, a_lo, ascore)
    a_level_md = fuzz.interp_membership(x_a, a_md, ascore)
    a_level_hi = fuzz.interp_membership(x_a, a_hi, ascore)

    s_level_lo = fuzz.interp_membership(x_s, s_lo, sscore)
    s_level_md = fuzz.interp_membership(x_s, s_md, sscore)
    s_level_hi = fuzz.interp_membership(x_s, s_hi, sscore)

    f_level_lo = fuzz.interp_membership(x_f, f_lo, fscore)
    f_level_md = fuzz.interp_membership(x_f, f_md, fscore)
    f_level_hi = fuzz.interp_membership(x_f, f_hi, fscore)

    active_rule1 = np.fmin(h_level_hi, np.fmin(s_level_hi, np.fmin(a_level_hi, f_level_hi)))
    active_rule2 = np.fmin(h_level_hi, np.fmin(s_level_hi, np.fmin(a_level_hi, f_level_lo)))
    active_rule3 = np.fmin(h_level_hi, np.fmin(s_level_hi, np.fmin(a_level_hi, f_level_md)))
    active_rule4 = np.fmin(h_level_hi, np.fmin(s_level_hi, np.fmin(a_level_lo, f_level_hi)))
    active_rule5 = np.fmin(h_level_hi, np.fmin(s_level_hi, np.fmin(a_level_lo, f_level_lo)))
    active_rule6 = np.fmin(h_level_hi, np.fmin(s_level_hi, np.fmin(a_level_lo, f_level_md)))
    active_rule7 = np.fmin(h_level_hi, np.fmin(s_level_hi, np.fmin(a_level_md, f_level_hi)))
    active_rule8 = np.fmin(h_level_hi, np.fmin(s_level_hi, np.fmin(a_level_md, f_level_lo)))
    active_rule9 = np.fmin(h_level_hi, np.fmin(s_level_hi, np.fmin(a_level_md, f_level_md)))
    active_rule10 = np.fmin(h_level_hi, np.fmin(s_level_lo, np.fmin(a_level_hi, f_level_hi)))
    active_rule11 = np.fmin(h_level_hi, np.fmin(s_level_lo, np.fmin(a_level_hi, f_level_lo)))
    active_rule12 = np.fmin(h_level_hi, np.fmin(s_level_lo, np.fmin(a_level_hi, f_level_md)))
    active_rule13 = np.fmin(h_level_hi, np.fmin(s_level_lo, np.fmin(a_level_lo, f_level_hi)))
    active_rule14 = np.fmin(h_level_hi, np.fmin(s_level_lo, np.fmin(a_level_lo, f_level_lo)))
    active_rule15 = np.fmin(h_level_hi, np.fmin(s_level_lo, np.fmin(a_level_lo, f_level_md)))
    active_rule16 = np.fmin(h_level_hi, np.fmin(s_level_lo, np.fmin(a_level_md, f_level_hi)))
    active_rule17 = np.fmin(h_level_hi, np.fmin(s_level_lo, np.fmin(a_level_md, f_level_lo)))
    active_rule18 = np.fmin(h_level_hi, np.fmin(s_level_lo, np.fmin(a_level_md, f_level_md)))
    active_rule19 = np.fmin(h_level_hi, np.fmin(s_level_md, np.fmin(a_level_hi, f_level_hi)))
    active_rule20 = np.fmin(h_level_hi, np.fmin(s_level_md, np.fmin(a_level_hi, f_level_lo)))
    active_rule21 = np.fmin(h_level_hi, np.fmin(s_level_md, np.fmin(a_level_hi, f_level_md)))
    active_rule22 = np.fmin(h_level_hi, np.fmin(s_level_md, np.fmin(a_level_lo, f_level_hi)))
    active_rule23 = np.fmin(h_level_hi, np.fmin(s_level_md, np.fmin(a_level_lo, f_level_lo)))
    active_rule24 = np.fmin(h_level_hi, np.fmin(s_level_md, np.fmin(a_level_lo, f_level_md)))
    active_rule25 = np.fmin(h_level_hi, np.fmin(s_level_md, np.fmin(a_level_md, f_level_hi)))
    active_rule26 = np.fmin(h_level_hi, np.fmin(s_level_md, np.fmin(a_level_md, f_level_lo)))
    active_rule27 = np.fmin(h_level_hi, np.fmin(s_level_md, np.fmin(a_level_md, f_level_md)))

    active_rule28 = np.fmin(h_level_lo, np.fmin(s_level_hi, np.fmin(a_level_hi, f_level_hi)))
    active_rule29 = np.fmin(h_level_lo, np.fmin(s_level_hi, np.fmin(a_level_hi, f_level_lo)))
    active_rule30 = np.fmin(h_level_lo, np.fmin(s_level_hi, np.fmin(a_level_hi, f_level_md)))
    active_rule31 = np.fmin(h_level_lo, np.fmin(s_level_hi, np.fmin(a_level_lo, f_level_hi)))
    active_rule32 = np.fmin(h_level_lo, np.fmin(s_level_hi, np.fmin(a_level_lo, f_level_lo)))
    active_rule33 = np.fmin(h_level_lo, np.fmin(s_level_hi, np.fmin(a_level_lo, f_level_md)))
    active_rule34 = np.fmin(h_level_lo, np.fmin(s_level_hi, np.fmin(a_level_md, f_level_hi)))
    active_rule35 = np.fmin(h_level_lo, np.fmin(s_level_hi, np.fmin(a_level_md, f_level_lo)))
    active_rule36 = np.fmin(h_level_lo, np.fmin(s_level_hi, np.fmin(a_level_md, f_level_md)))
    active_rule37 = np.fmin(h_level_lo, np.fmin(s_level_lo, np.fmin(a_level_hi, f_level_hi)))
    active_rule38 = np.fmin(h_level_lo, np.fmin(s_level_lo, np.fmin(a_level_hi, f_level_lo)))
    active_rule39 = np.fmin(h_level_lo, np.fmin(s_level_lo, np.fmin(a_level_hi, f_level_md)))
    active_rule40 = np.fmin(h_level_lo, np.fmin(s_level_lo, np.fmin(a_level_lo, f_level_hi)))
    active_rule41 = np.fmin(h_level_lo, np.fmin(s_level_lo, np.fmin(a_level_lo, f_level_lo)))
    active_rule42 = np.fmin(h_level_lo, np.fmin(s_level_lo, np.fmin(a_level_lo, f_level_md)))
    active_rule43 = np.fmin(h_level_lo, np.fmin(s_level_lo, np.fmin(a_level_md, f_level_hi)))
    active_rule44 = np.fmin(h_level_lo, np.fmin(s_level_lo, np.fmin(a_level_md, f_level_lo)))
    active_rule45 = np.fmin(h_level_lo, np.fmin(s_level_lo, np.fmin(a_level_md, f_level_md)))
    active_rule46 = np.fmin(h_level_lo, np.fmin(s_level_md, np.fmin(a_level_hi, f_level_hi)))
    active_rule47 = np.fmin(h_level_lo, np.fmin(s_level_md, np.fmin(a_level_hi, f_level_lo)))
    active_rule48 = np.fmin(h_level_lo, np.fmin(s_level_md, np.fmin(a_level_hi, f_level_md)))
    active_rule49 = np.fmin(h_level_lo, np.fmin(s_level_md, np.fmin(a_level_lo, f_level_hi)))
    active_rule50 = np.fmin(h_level_lo, np.fmin(s_level_md, np.fmin(a_level_lo, f_level_lo)))
    active_rule51 = np.fmin(h_level_lo, np.fmin(s_level_md, np.fmin(a_level_lo, f_level_md)))
    active_rule52 = np.fmin(h_level_lo, np.fmin(s_level_md, np.fmin(a_level_md, f_level_hi)))
    active_rule53 = np.fmin(h_level_lo, np.fmin(s_level_md, np.fmin(a_level_md, f_level_lo)))
    active_rule54 = np.fmin(h_level_lo, np.fmin(s_level_md, np.fmin(a_level_md, f_level_md)))
    active_rule55 = np.fmin(h_level_md, np.fmin(s_level_md, np.fmin(a_level_md, f_level_md)))
    active_rule56 = np.fmin(h_level_md, np.fmin(s_level_hi, np.fmin(a_level_hi, f_level_hi)))
    active_rule57 = np.fmin(h_level_md, np.fmin(s_level_hi, np.fmin(a_level_hi, f_level_lo)))
    active_rule58 = np.fmin(h_level_md, np.fmin(s_level_hi, np.fmin(a_level_hi, f_level_md)))
    active_rule59 = np.fmin(h_level_md, np.fmin(s_level_hi, np.fmin(a_level_lo, f_level_hi)))
    active_rule60 = np.fmin(h_level_md, np.fmin(s_level_hi, np.fmin(a_level_lo, f_level_lo)))
    active_rule61 = np.fmin(h_level_md, np.fmin(s_level_hi, np.fmin(a_level_lo, f_level_md)))
    active_rule62 = np.fmin(h_level_md, np.fmin(s_level_hi, np.fmin(a_level_md, f_level_hi)))
    active_rule63 = np.fmin(h_level_md, np.fmin(s_level_hi, np.fmin(a_level_md, f_level_lo)))
    active_rule64 = np.fmin(h_level_md, np.fmin(s_level_hi, np.fmin(a_level_md, f_level_md)))
    active_rule65 = np.fmin(h_level_md, np.fmin(s_level_lo, np.fmin(a_level_hi, f_level_hi)))
    active_rule66 = np.fmin(h_level_md, np.fmin(s_level_lo, np.fmin(a_level_hi, f_level_lo)))
    active_rule67 = np.fmin(h_level_md, np.fmin(s_level_lo, np.fmin(a_level_hi, f_level_md)))
    active_rule68 = np.fmin(h_level_md, np.fmin(s_level_lo, np.fmin(a_level_lo, f_level_hi)))
    active_rule69 = np.fmin(h_level_md, np.fmin(s_level_lo, np.fmin(a_level_lo, f_level_lo)))
    active_rule70 = np.fmin(h_level_md, np.fmin(s_level_lo, np.fmin(a_level_lo, f_level_md)))
    active_rule71 = np.fmin(h_level_md, np.fmin(s_level_lo, np.fmin(a_level_md, f_level_hi)))
    active_rule72 = np.fmin(h_level_md, np.fmin(s_level_lo, np.fmin(a_level_md, f_level_lo)))
    active_rule73 = np.fmin(h_level_md, np.fmin(s_level_lo, np.fmin(a_level_md, f_level_md)))
    active_rule74 = np.fmin(h_level_md, np.fmin(s_level_md, np.fmin(a_level_hi, f_level_hi)))
    active_rule75 = np.fmin(h_level_md, np.fmin(s_level_md, np.fmin(a_level_hi, f_level_lo)))
    active_rule76 = np.fmin(h_level_md, np.fmin(s_level_md, np.fmin(a_level_hi, f_level_md)))
    active_rule77 = np.fmin(h_level_md, np.fmin(s_level_md, np.fmin(a_level_lo, f_level_hi)))
    active_rule78 = np.fmin(h_level_md, np.fmin(s_level_md, np.fmin(a_level_lo, f_level_lo)))
    active_rule79 = np.fmin(h_level_md, np.fmin(s_level_md, np.fmin(a_level_lo, f_level_md)))
    active_rule80 = np.fmin(h_level_md, np.fmin(s_level_md, np.fmin(a_level_md, f_level_hi)))
    active_rule81 = np.fmin(h_level_md, np.fmin(s_level_md, np.fmin(a_level_md, f_level_lo)))
    h_final = np.fmax(active_rule14, np.fmax(active_rule15, np.fmax(active_rule17, np.fmax(active_rule18,
                                                                                           np.fmax(active_rule23,
                                                                                                   np.fmax(
                                                                                                       active_rule24,
                                                                                                       np.fmax(
                                                                                                           active_rule26,
                                                                                                           np.fmax(
                                                                                                               active_rule27,
                                                                                                               active_rule69))))))))
    op_activation_ha = np.fmin(h_final, op_hap)
    print(h_final)
    print(op_activation_ha)

    a_final = np.fmax(active_rule38, np.fmax(active_rule39, np.fmax(active_rule44, np.fmax(active_rule47,
                                                                                           np.fmax(active_rule48,
                                                                                                   np.fmax(
                                                                                                       active_rule66,
                                                                                                       np.fmax(
                                                                                                           active_rule67,
                                                                                                           np.fmax(
                                                                                                               active_rule75,
                                                                                                               active_rule76))))))))
    op_activation_an = np.fmin(a_final, op_ang)
    print(a_final)
    print(op_activation_an)
    s_final = np.fmax(active_rule32, np.fmax(active_rule33, np.fmax(active_rule35, np.fmax(active_rule36,
                                                                                           np.fmax(active_rule50,
                                                                                                   np.fmax(
                                                                                                       active_rule60,
                                                                                                       np.fmax(
                                                                                                           active_rule61,
                                                                                                           np.fmax(
                                                                                                               active_rule63,
                                                                                                               active_rule64))))))))
    op_activation_sa = np.fmin(s_final, op_sad)
    print(s_final)
    print(op_activation_sa)
    f_final = np.fmax(active_rule80, np.fmax(active_rule77, np.fmax(active_rule71, np.fmax(active_rule68,
                                                                                           np.fmax(active_rule52,
                                                                                                   np.fmax(
                                                                                                       active_rule49,
                                                                                                       np.fmax(
                                                                                                           active_rule43,
                                                                                                           np.fmax(
                                                                                                               active_rule42,
                                                                                                               active_rule40))))))))
    op_activation_fe = np.fmin(f_final, op_fea)
    print(f_final)
    print(op_activation_fe)

    op0 = np.zeros_like(x_op)

    # Aggregate all three output membership functions together
    aggregated = np.fmax(op_activation_ha,
                         np.fmax(op_activation_sa, np.fmax(op_activation_fe, op_activation_an)))
    print(aggregated)
    op = 0
    op = fuzz.defuzz(x_op, aggregated, 'mom')
    output = round(op, 2)
    op_activation = fuzz.interp_membership(x_op, aggregated, op)  # for plot
    print(op)
    final_emotion=""
    final2_emotion=""
    if 0 < (output) < 2.5:
        final_emotion = "happy"
        print("\nOutput after Defuzzification: happy")


    elif 2.6 < (output) < 5.0:
        print("\nOutput after Defuzzification: sad")
        final_emotion = "sad"

    elif 5.1 < (output) < 7.5:
        print("\nOutput after Defuzzification: fear")
        final_emotion = "fear"
    elif 7.6 < (output) < 10:
        final_emotion = "angry"
        print("\nOutput after Defuzzification: angry")
    print(output)
    if final_emotion:
        return final_emotion, output
    op = fuzz.defuzz(x_op, aggregated, 'bisector')
    output = round(op, 2)
    op_activation = fuzz.interp_membership(x_op, aggregated, op)  # for plot
    print(op)
    if 0 < (output) < 2.5:
        final2_emotion = "happy"
        print("\nOutput after Defuzzification: happy")


    elif 2.6 < (output) < 5.0:
        final2_emotion = "sad"
        print("\nOutput after Defuzzification: sad")


    elif 5.1 < (output) < 7.5:
        final2_emotion = "fear"
        print("\nOutput after Defuzzification: fear")

    elif 7.6 < (output) < 10:
        final2_emotion = "angry"
        print("\nOutput after Defuzzification: angry")
    if final_emotion==final2_emotion:
        return final_emotion
    if final2_emotion:
        return final2_emotion, output

    return "no emotion detected"


def intensity(m_score, ar_score):
    m_level_lo = fuzz.interp_membership(x_m, m_lo, m_score)
    m_level_md = fuzz.interp_membership(x_m, m_md, m_score)
    m_level_hi = fuzz.interp_membership(x_m, m_hi, m_score)

    ar_level_lo = fuzz.interp_membership(x_ar, ar_lo, ar_score)
    ar_level_md = fuzz.interp_membership(x_ar, ar_md, ar_score)
    ar_level_hi = fuzz.interp_membership(x_ar, ar_hi, ar_score)

    act_rule1 = np.fmin(ar_level_hi, m_level_hi)
    act_rule2 = np.fmin(ar_level_hi, m_level_md)
    act_rule3 = np.fmin(ar_level_hi, m_level_lo)
    act_rule4 = np.fmin(ar_level_md, m_level_hi)
    act_rule5 = np.fmin(ar_level_md, m_level_md)
    act_rule6 = np.fmin(ar_level_md, m_level_lo)
    act_rule7 = np.fmin(ar_level_lo, m_level_hi)
    act_rule8 = np.fmin(ar_level_lo, m_level_md)
    act_rule9 = np.fmin(ar_level_lo, m_level_lo)

    el_final = np.fmax(act_rule9, act_rule6)
    op_activation_1 = np.fmin(el_final, op_1)

    l_final = np.fmax(act_rule3, act_rule8)
    op_activation_2 = np.fmin(el_final, op_2)

    m_final = np.fmax(act_rule5, act_rule8)
    op_activation_3 = np.fmin(m_final, op_3)

    h1_final = np.fmax(act_rule7, act_rule2)
    op_activation_4 = np.fmin(h1_final, op_4)

    eh_final = np.fmax(act_rule1, act_rule4)
    op_activation_5 = np.fmin(eh_final, op_5)

    op0 = np.zeros_like(x_op)

    # Aggregate all output membership functions together
    aggregated_ei = np.fmax(op_activation_1,
                            np.fmax(op_activation_2,
                                    np.fmax(op_activation_3, np.fmax(op_activation_4, op_activation_5))))
    # print(aggregated_ei)

    op = fuzz.defuzz(x_op_ei, aggregated_ei, 'bisector')
    output = round(op, 2)
    op_activation = fuzz.interp_membership(x_op_ei, aggregated_ei, op)  # for plot
    print(op)

    if 0 < (output) <= 4:
        intensity = "low"
        print("\nOutput after Defuzzification: low")
    elif 4.01 <= (output) <= 4.50:
        intensity = "medium"
        print("\nOutput after Defuzzification: moderate")
    elif 4.51 <= (output) <= 10:
        intensity = "high"
        print("\nOutput after Defuzzification: high")
    return output,intensity


def intensity_fuzzification(sentence_array, hscore, fscore, ascore, sscore):
    df = pd.read_csv("nrc_vad.csv")
    df.head()
    arousal_vector = np.zeros((len(sentence_array), 4))
    i = 0
    for word in sentence_array:
        emotion = NRCLex(word)
        if "sadness" in emotion.affect_list:
            wi = 0
            for words in df["Word"]:
                if word == words:
                    arousal_vector[i][0] = df["Arousal"][wi]
                    break
                if Lemmatizer.lemmatize(word) == words:
                    arousal_vector[i][0] = df["Arousal"][wi]
                    break
        if "joy" in emotion.affect_list:
            wi = 0
            for words in df["Word"]:
                if word == words:
                    arousal_vector[i][1] = df["Arousal"][wi]
                    break
                if Lemmatizer.lemmatize(word) == words:
                    arousal_vector[i][1] = df["Arousal"][wi]
                    break
        if "fear" in emotion.affect_list:
            wi = 0
            for words in df["Word"]:
                if word == words:
                    arousal_vector[i][2] = df["Arousal"][wi]
                    break
                if Lemmatizer.lemmatize(word) == words:
                    arousal_vector[i][2] = df["Arousal"][wi]
                    break
        if "anger" in emotion.affect_list:
            wi = 0
            for words in df["Word"]:
                if word == words:
                    arousal_vector[i][3] = df["Arousal"][wi]
                    break
                if Lemmatizer.lemmatize(word) == words:
                    arousal_vector[i][3] = df["Arousal"][wi]
                    break

            wi = wi + 1
        i = i + 1
    print(arousal_vector)
    sa_score = 0
    ha_score = 0
    fa_score = 0
    aa_score = 0
    sa_nos = 0
    ha_nos = 0
    fa_nos = 0
    aa_nos = 0
    for i in range(len(sentence_array)):
        if (arousal_vector[i][0] != 0.0):
            sa_score = sa_score + arousal_vector[i][0]
            sa_nos = sa_nos + 1
        if (arousal_vector[i][1] != 0.0):
            ha_score = ha_score + arousal_vector[i][1]
            ha_nos = ha_nos + 1
        if (arousal_vector[i][2] != 0.0):
            fa_score = fa_score + arousal_vector[i][2]
            fa_nos = fa_nos + 1
        if (arousal_vector[i][3] != 0.0):
            aa_score = aa_score + arousal_vector[i][3]
            aa_nos = aa_nos + 1
    if (sa_nos != 0):
        sa_score = sa_score / sa_nos
    if (ha_nos != 0):
        ha_score = ha_score / ha_nos
    if (aa_nos != 0):
        aa_score = aa_score / aa_nos
    if (fa_nos != 0):
        fa_score = fa_score / fa_nos

    print(aa_score)
    print(ha_score)
    print(sa_score)
    print(fa_score)
    hascore = ha_score
    if ha_score == 0:
        hascore = 0.01
    aascore = aa_score
    if aa_score == 0:
        aascore = 0.01
    sascore = sa_score
    if sa_score == 0:
        sascore = 0.01
    fascore = fa_score
    if fa_score == 0:
        fascore = 0.01
    print("happy")
    happy_intensity,h_intensity = intensity(hscore, hascore)
    print("angry")
    angry_intensity,a_intensity = intensity(ascore, aascore)
    print("sad")
    sad_intensity,s_intensity = intensity(sscore, sascore)
    print("fear")
    fear_intensity,f_intensity = intensity(fscore, fascore)

    return happy_intensity, angry_intensity, sad_intensity, fear_intensity, h_intensity, a_intensity, s_intensity, f_intensity


def emoji_handling(tweet):
    df = pd.read_csv("EmoTag1200-scores.csv")

    emo_vector_size = 0
    for i in tweet:
        for j in df["name"]:
            if emoji.demojize(i) == ":" + j + ":":
                emo_vector_size = emo_vector_size + 1

    emoji_score = np.zeros((emo_vector_size, 4))
    for i in tweet:
        emoji_index = 0
        k = 0
        for j in df["name"]:
            if emoji.demojize(i) == ":" + j + ":":
                emoji_score[k][0] = df["sadness"][emoji_index]
                emoji_score[k][1] = df["joy"][emoji_index]
                emoji_score[k][2] = df["fear"][emoji_index]
                emoji_score[k][3] = df["anger"][emoji_index]
                k = k + 1
                break
            emoji_index = emoji_index + 1

    return emoji_score

# --------------------------------------APPLICATION--------------------------------------------------------


@app.route("/")
def index():
    return render_template('index.html')
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/search", methods=["GET", "POST"])
def tweet_emotion_detection():
    if request.method == "POST":
        raw = request.form.get('search')
        search_tweet = BeautifulSoup(raw, 'html.parser').get_text()
    t = []
    print(raw)
    tweets = tweepy.Cursor(api.search, q=search_tweet, lang="en", tweet_mode='extended').items(15)
    #tweets = api.search(search_tweet, tweet_mode='extended')
    final_result = []
    for twt in tweets:
        i = twt.full_text
        c = []
        c.append(i)
        predicted_emotion = None
        preprocessed_text = tweet_preprocessing(i)
        print("tweet : " + i)
        amusement = 0
        anger = 0
        awe = 0
        contentment = 0
        disgust = 0
        excitement = 0
        fear = 0
        sadness=0
        if twt.entities.get('media', []):
            media = twt.entities.get('media', [])
            image_url = media[0]['media_url']
            print("image url " + image_url)
            res = requi.urlopen(image_url).read()
            transformed_image = Image.open(BytesIO(res))
            image_for_caption = send_image_for_reshape(transformed_image)
            generated_caption = beam_search_predictions(image_for_caption, beam_index = 3)
            cosine_value = cosine_similarity(generated_caption, i)
            print(generated_caption)

            if cosine_value:
                predicted_emotion, amusement, anger, awe, contentment, disgust, excitement, fear, sadness = image_emotion_prediction(transformed_image)
            else:
                predicted_emotion = None
            print(predicted_emotion)

        emoji_score_vector = emoji_handling(i)
        sad_score, fear_score, happy_score, angry_score, sentence_array = scoring_tweets(preprocessed_text,
                                                                                         emoji_score_vector)
        detected_emotion, output = fuzzification(sad_score, fear_score, happy_score, angry_score)
        happy_intensity, angry_intensity, sad_intensity, fear_intensity, h_intensity, a_intensity, s_intensity, f_intensity = intensity_fuzzification(
            sentence_array, happy_score, fear_score, angry_score, sad_score)
        c.append(detected_emotion)
        c.append(happy_intensity)
        c.append(angry_intensity)
        c.append(sad_intensity)
        c.append(fear_intensity)
        c.append(h_intensity)
        c.append(a_intensity)
        c.append(s_intensity)
        c.append(f_intensity)
        c.append(output)
        c.append(predicted_emotion)
        c.append(amusement)
        c.append(anger)
        c.append(awe)
        c.append(contentment)
        c.append(disgust)
        c.append(excitement)
        c.append(fear)
        c.append(sadness)

        final_result.append(c)

    return render_template("search_tweet_output.html", search_twt=final_result)


@app.route("/search_text", methods=["GET", "POST"])
def text_emotion_detection():
    if request.method == "POST":
        c_text = request.form.get('custom_text')
        raw = BeautifulSoup(c_text, 'html.parser').get_text()
        print(raw)
        t = []
        t.append(raw)
        preprocessed_text = tweet_preprocessing(raw)
        emoji_score_vector = emoji_handling(raw)
        sad_score, fear_score, happy_score, angry_score, sentence_array = scoring_tweets(preprocessed_text,  emoji_score_vector)
        detected_emotion,output = fuzzification(sad_score, fear_score, happy_score, angry_score)
        happy_intensity, angry_intensity, sad_intensity, fear_intensity, h_intensity, a_intensity, s_intensity, f_intensity = intensity_fuzzification(sentence_array, happy_score, fear_score, angry_score, sad_score)
        t.append(detected_emotion)
        t.append(happy_intensity)
        t.append(angry_intensity)
        t.append(sad_intensity)
        t.append(fear_intensity)
        t.append(h_intensity)
        t.append(a_intensity)
        t.append(s_intensity)
        t.append(f_intensity)
        t.append(output)
    return render_template("custom_tweet_output.html", custom_twt=t)


app.run(debug=True)
