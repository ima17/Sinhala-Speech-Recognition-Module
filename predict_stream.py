import numpy as np
from data_generator import AudioGenerator
from keras import backend as K
import tensorflow as tf
from utils import int_sequence_to_text
from IPython.display import Audio
import os


# import mysql.connector
#
# myDb = mysql.connector.connect(host="localhost", user="root", password="jayanieboga@1995",
#                                database="speech_recognition")
# print(myDb)
#
# if myDb:
#     print("Connection successful")
#
# else:
#     print("Connection failed")
# myCursor = myDb.cursor()


def get_predictions2(input_to_softmax, model_path):
    """ Print a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        input_to_softmax (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    data_gen = AudioGenerator(spectrogram=False)
    data_gen.load_train_data()
    # myCursor.execute("Select wav_file from wav_files")
    # audio_path1 = myCursor.fetchone()
    audio_path1 = "D:\\level4_sem1\\project_interim2\\wavFiles\\output1.wav"
    # audio_path2 = "D:\\level4_sem1\\project_interim2\\wavFiles\\output2.wav"
    # audio_path3 = "D:\\level4_sem1\\project_interim2\\wavFiles\\output3.wav"
    # audio_path4 = "D:\\level4_sem1\\project_interim2\\wavFiles\\output4.wav"
    # print(data_gen.featurize(audio_path))

    data_point1 = data_gen.normalize(data_gen.featurize(audio_path1))
    # data_point2 = data_gen.normalize(data_gen.featurize(audio_path2))
    # data_point3 = data_gen.normalize(data_gen.featurize(audio_path3))
    # data_point4 = data_gen.normalize(data_gen.featurize(audio_path4))

    # obtain and decode the acoustic model's predictions

    # print(np.expand_dims(data_point, axis=0))
    input_to_softmax.load_weights(model_path)

    prediction1 = input_to_softmax.predict(np.expand_dims(data_point1, axis=0))
    output_length1 = [input_to_softmax.output_length(data_point1.shape[0])]
    # prediction2 = input_to_softmax.predict(np.expand_dims(data_point2, axis=0))
    # output_length2 = [input_to_softmax.output_length(data_point2.shape[0])]
    # prediction3 = input_to_softmax.predict(np.expand_dims(data_point3, axis=0))
    # output_length3 = [input_to_softmax.output_length(data_point3.shape[0])]
    # prediction4 = input_to_softmax.predict(np.expand_dims(data_point4, axis=0))
    # output_length4 = [input_to_softmax.output_length(data_point4.shape[0])]
    # print(prediction)
    # print(output_length)

    # pred_ints = ((prediction, output_length)[0][0]+1).flatten().tolist()

    pred_ints1 = (tf.keras.backend.eval(
        tf.keras.backend.ctc_decode(prediction1, output_length1)[0][0]) + 1).flatten().tolist()
    # pred_ints2 = (tf.keras.backend.eval(
    #     tf.keras.backend.ctc_decode(prediction2, output_length2)[0][0]) + 1).flatten().tolist()
    # pred_ints3 = (tf.keras.backend.eval(
    #     tf.keras.backend.ctc_decode(prediction3, output_length3)[0][0]) + 1).flatten().tolist()
    # pred_ints4 = (tf.keras.backend.eval(
    #     tf.keras.backend.ctc_decode(prediction4, output_length4)[0][0]) + 1).flatten().tolist()

    # play the audio file, and display the true and predicted transcriptions
    print('-' * 80)
    print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints1)))
    print('-' * 80)
    # print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints2)))
    # print('-' * 80)
    # print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints3)))
    # print('-' * 80)
    # print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints4)))
    os.remove("D:/level4_sem1/project_interim2/output/transcriptions.txt")

    L1 = [''.join(int_sequence_to_text(pred_ints1))]
    # L2 = [''.join(int_sequence_to_text(pred_ints2))]
    # L3 = [''.join(int_sequence_to_text(pred_ints3))]
    # L4 = [''.join(int_sequence_to_text(pred_ints4))]
    L5 = [" "]
    file2 = open("D:/level4_sem1/project_interim2/output/transcriptions.txt", "a", encoding="utf-8")
    file2.writelines(L1)
    file2.writelines(L5)
    # file2.writelines(L2)
    file2.writelines(L5)
    # file2.writelines(L3)
    file2.writelines(L5)
    # file2.writelines(L4)
    file2.close()
    os.system("start notepad.exe D:/level4_sem1/project_interim2/output/transcriptions.txt")



