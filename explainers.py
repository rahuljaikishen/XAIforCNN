from lime import lime_image
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from common import get_predictions, evaluate_pixels, preprocess_img
import cv2
from tensorflow.keras.models import Model


def get_explanations(explanation_type, img, model, model_name):
    if explanation_type == 'LIME':
        x = preprocess_input(img)
        # x = preprocess_img(img)
        heatmap = get_lime_heatmap(x, model)
    elif explanation_type == 'CIE_INSPIRATION':
        heatmap = get_cie_insp_heatmap(model_name, model, img)
    elif explanation_type == 'FEATURE_EXTRACTION_LAST_LAYER':
        x = preprocess_img(img)
        heatmap = get_last_layer_explanations(model, x)

    return heatmap


def get_lime_heatmap(x, model):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(x.astype('double'), model.predict, top_labels=5, hide_color=0,
                                             num_samples=10)
    ind = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    return heatmap


def get_cie_insp_heatmap(model_name, model, img, window_size=56, row_lim=168, col_lim=168):
    prediction, prediction_score, prediction_type = get_predictions(model_name, model, img)
    if model_name == 'VGG16':
        mask = np.zeros(shape=(224, 224))
    else:
        mask = np.zeros(shape=(229, 229))
    row_start = 0
    frame_no = 0
    while row_start <= row_lim:
        column_start = 0
        while column_start <= col_lim:
            frame = img.copy()
            top_prediction, top_pred_score, prediction_type = evaluate_pixels(row_start, row_start + window_size,
                                                                              column_start,
                                                                              column_start + window_size, frame,
                                                                              model_name, model, prediction)
            print('Prediction for block ', frame_no + 1, ' - ', top_prediction, top_pred_score, prediction_type)
            diff = top_pred_score - prediction_score
            # normalize_diff =
            mask[row_start:row_start + window_size, column_start:column_start + window_size] = diff
            column_start = column_start + window_size
            frame_no = frame_no + 1
        row_start = row_start + window_size
    return mask


def get_last_layer_explanations(model, img):
    ixs = [15]
    outputs = [model.layers[i].output for i in ixs]
    model_new = Model(inputs=model.inputs, outputs=outputs)
    feature_maps = model_new.predict(img)
    square = 8
    for fmap in feature_maps:
        ix = 1
        for _ in range(square):
            for _ in range(square):
                if 'img1' in locals():
                    img1 = cv2.addWeighted(img1, 0.3, fmap[:, :, ix - 1], 0.7, 0)
                else:
                    img1 = fmap[:, :, ix - 1]
                ix += 1
    return img1
