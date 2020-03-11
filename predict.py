import numpy as np
from data_generator import AudioGenerator
from keras import backend as K
import tensorflow as tf
from utils import int_sequence_to_text
from IPython.display import Audio, display
import IPython
import os


def get_predictions(index, partition, input_to_softmax, model_path):
    """ Print a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        input_to_softmax (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    # load the train and test data
    data_gen = AudioGenerator(spectrogram=False)
    data_gen.load_train_data()
    data_gen.load_test_data()

    # obtain the true transcription and the audio features 
    if partition == 'test':
        transcr = data_gen.test_texts[index]
        audio_path = data_gen.test_audio_paths[index]
        print(audio_path)
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    elif partition == 'train':
        transcr = data_gen.train_texts[index]
        audio_path = data_gen.train_audio_paths[index]
        print(audio_path)
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    else:
        raise Exception('Invalid partition!  Must be "train" or "validation"')

    # obtain and decode the acoustic model's predictions

    # print(np.expand_dims(data_point, axis=0))
    input_to_softmax.load_weights(model_path)
    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    output_length = [input_to_softmax.output_length(data_point.shape[0])]
    print(prediction)
    print(output_length)

    # pred_ints = ((prediction, output_length)[0][0]+1).flatten().tolist()

    pred_ints = (tf.keras.backend.eval(tf.keras.backend.ctc_decode(prediction, output_length)[0][0]) + 1).flatten() \
        .tolist()

    # play the audio file, and display the true and predicted transcriptions
    print('-' * 80)
    print(audio_path)
    IPython.display.Audio(audio_path, autoplay=True)
    print('True transcription:\n' + '\n' + transcr)
    print('-' * 80)
    print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints)))
    print('-' * 80)
    os.remove("D:/level4_sem1/project_interim2/output/transcriptions.txt")
    T = ["True Transcription: "]
    L1 = [transcr]
    L3 = ["         "]
    T1 = ["Predicted Transcription: "]
    L2 = [''.join(int_sequence_to_text(pred_ints))]

    file2 = open("D:/level4_sem1/project_interim2/output/transcriptions.txt", "a", encoding="utf-8")
    file2.writelines(T)
    file2.writelines(L1)
    file2.writelines(L3)
    file2.writelines(T1)
    file2.writelines(L2)
    file2.close()
