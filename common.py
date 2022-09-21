from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.vgg16 import decode_predictions as decode_predictions_vgg16
import numpy as np


def get_predictions(model_name, model, img_arr, last_prediction=''):
    x = preprocess_img(img_arr)
    predictions = model.predict(x)
    if model_name == 'VGG16':
        predictions_decoded = decode_predictions_vgg16(predictions)[0]
    elif model_name == 'Inception':
        predictions_decoded = decode_predictions(predictions)[0]
    top_prediction, top_pred_score, prediction_type = get_top_scores(predictions_decoded, last_prediction)
    return top_prediction, top_pred_score, prediction_type


def get_top_scores(predictions, last_prediction):
    top_prediction, top_pred_score, prediction_type = last_prediction, 0, 0
    # 0 - initial state
    # 1 - image modified and prediction still in first position
    # 2 - image modified and prediction no more in first position
    # 3 - first image prediction
    for index, x in enumerate(predictions):
        label = x[1]
        score = x[2]
        if last_prediction != '' and last_prediction == label:
            if index == 0:
                top_prediction, top_pred_score, prediction_type = label, score, 1
            else:
                top_prediction, top_pred_score, prediction_type = label, score, 2
        else:
            top_prediction, top_pred_score, prediction_type = label, score, 3

        if prediction_type != 0:
            break
    return top_prediction, top_pred_score, prediction_type


def preprocess_img(img_arr):
    x = np.expand_dims(img_arr, axis=0)
    x = preprocess_input(x)
    return x


def evaluate_pixels(row_start, row_to, column_start, column_to, frame, model_name, model, prediction):
    # blur = cv2.GaussianBlur(frame[row_start:row_to,column_start:column_to],(3,3),0)
    frame[row_start:row_to, column_start:column_to] = (0, 0, 0)
    top_prediction, top_pred_score, prediction_type = get_predictions(model_name, model, frame, prediction)
    return top_prediction, top_pred_score, prediction_type
