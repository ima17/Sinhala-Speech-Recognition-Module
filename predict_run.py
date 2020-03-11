from predict import get_predictions
from predict_stream import get_predictions2
from sample_models import *
#
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
#                           output_dim=48,
#                           activation='relu',
#                           cell=GRU,
#                           dropout_rate=0.01,
#                           number_of_layers=2)

# BEST model_end4
# model_end6 is also good (better than model_end4)
# get_predictions(index=21,
#                 partition='test',
#                 input_to_softmax=model_end10,
#                 model_path='results/model_end10.h5')
#
get_predictions2(input_to_softmax=model_end12,
                 model_path='results/model_end12.h5')
