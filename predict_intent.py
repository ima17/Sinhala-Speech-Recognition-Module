import pandas as pd
import numpy as np
import pickle

import playsound
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from pathlib import Path
import cv2



# For reproducibility
np.random.seed(1237)


labels = np.array(['ask.currency.value', 'greetings.words', 'not.included'])

# load our saved model
model = load_model('my_model.h5')

# load tokenizer
tokenizer = Tokenizer()
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

f = open("D:/level4_sem1/project_interim2/output/transcriptions.txt", "r", encoding="ISO-8859-1")
x_data = f.readline()
print(x_data)

x_data_series = pd.Series(x_data)
x_tokenized = tokenizer.texts_to_matrix(x_data_series, mode='tfidf')

i = 0
for x_t in x_tokenized:
    prediction = model.predict(np.array([x_t]))
    predicted_label = labels[np.argmax(prediction[0])]
    print("File ->", "Predicted label: " + predicted_label)
    i += 1
    if predicted_label == "ask.currency.value":
        print("INTENT IS THERE")
        playsound.playsound('D:/level4_sem1/project_interim2/mp3/good2.mp3', True)
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)

        if vc.isOpened():  # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False

        while rval:
            cv2.imshow("preview", frame)
            rval, frame = vc.read()
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
        vc.release()
        cv2.destroyWindow("preview")

    else:
        print("INTENT IS NOT THERE")
        playsound.playsound('D:/level4_sem1/project_interim2/mp3/bad2.mp3', True)
