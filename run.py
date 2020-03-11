# import NN architectures for speech recognition
from sample_models import *
# import function for training acoustic model
from train_utils import train_model

# model_end = final_model(input_dim=161,
#                         filters=200,
#                         kernel_size=11,
#                         conv_stride=2,
#                         conv_border_mode='valid',
#                         units=250,
#                         activation='relu',
#                         cell=GRU,
#                         dropout_rate=1,
#                         number_of_layers=2)
#
# model_end2 = final_model(input_dim=161,
#                          filters=200,
#                          kernel_size=11,
#                          conv_stride=2,
#                          conv_border_mode='valid',
#                          units=250,
#                          activation='relu',
#                          cell=GRU,
#                          dropout_rate=0.5,
#                          number_of_layers=2)
#
# model_end3 = final_model(input_dim=13,
#                          filters=200,
#                          kernel_size=11,
#                          conv_stride=2,
#                          conv_border_mode='valid',
#                          units=250,
#                          activation='relu',
#                          cell=GRU,
#                          dropout_rate=1,
#                          number_of_layers=2)
#
# model_end4 = final_model(input_dim=13,
#                          filters=200,
#                          kernel_size=5,
#                          conv_stride=2,
#                          conv_border_mode='valid',
#                          units=250,
#                          activation='relu',
#                          cell=GRU,
#                          dropout_rate=1,
#                          number_of_layers=2)
#
# model_end5 = final_model(input_dim=13,
#                          filters=200,
#                          kernel_size=3,
#                          conv_stride=2,
#                          conv_border_mode='valid',
#                          units=250,
#                          activation='relu',
#                          cell=GRU,
#                          dropout_rate=1,
#                          number_of_layers=2)
#
# model_end6 = final_model(input_dim=13,
#                          filters=200,
#                          kernel_size=3,
#                          conv_stride=2,
#                          conv_border_mode='valid',
#                          units=250,
#                          activation='relu',
#                          cell=GRU,
#                          dropout_rate=1,
#                          number_of_layers=2)
#
# model_end10 = final_model(input_dim=13,
#                           filters=200,
#                           kernel_size=3,
#                           conv_stride=2,
#                           conv_border_mode='valid',
#                           units=250,
#                           activation='relu',
#                           cell=GRU,
#                           dropout_rate=0.2,
#                           number_of_layers=2)

# model_end11 = final_model(input_dim=13,
#                           filters=200,
#                           kernel_size=3,
#                           conv_stride=2,
#                           conv_border_mode='valid',
#                           units=250,
#                           activation='relu',
#                           cell=GRU,
#                           dropout_rate=0.2,
#                           number_of_layers=2)

model_end12 = final_model(input_dim=13,
                          filters=200,
                          kernel_size=3,
                          conv_stride=2,
                          conv_border_mode='valid',
                          units=250,
                          activation='relu',
                          cell=GRU,
                          dropout_rate=0.1,
                          number_of_layers=2)

# model_end13 = final_model(input_dim=13,
#                           filters=200,
#                           kernel_size=3,
#                           conv_stride=2,
#                           conv_border_mode='valid',
#                           units=250,
#                           activation='relu',
#                           cell=GRU,
#                           dropout_rate=0.1,
#                           number_of_layers=2)

# model_end13 = final_model(input_dim=13,
#                           filters=200,
#                           kernel_size=3,
#                           conv_stride=2,
#                           conv_border_mode='valid',
#                           units=250,
#                           output_dim=48,
#                           activation='relu',
#                           cell=GRU,
#                           dropout_rate=0.01,
#                           number_of_layers=2)

train_model(input_to_softmax=model_end12,
            pickle_path='model_end12.pickle',
            save_model_path='model_end12.h5',
            spectrogram=False)
