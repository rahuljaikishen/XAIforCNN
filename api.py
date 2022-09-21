import uvicorn
from fastapi import FastAPI, UploadFile, Request
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from io import BytesIO
from typing import Union
import json
from explainers import get_explanations
from common import get_predictions
from tensorflow.keras.applications import inception_v3 as inc_net
import cv2
app = FastAPI(title='Tensorflow FastAPI')


def get_img_arr(img, model_name, input_shape):
    image = Image.open(BytesIO(img))
    image = image.convert("RGB")
    image = image.resize(input_shape)
    image = img_to_array(image)
    return image

@app.post("/explain")
async def explain(request: Request, file: Union[UploadFile, None] = None):
    form = await request.form()
    contents = form["upload_file"].file
    data = contents.read()
    model_name = form['model_name']
    if model_name == 'VGG16':
        input_shape = (224, 224)
        img_processed = get_img_arr(data, model_name, input_shape)
        model = VGG16()
    else:
        input_shape = (229, 229)
        img_processed = get_img_arr(data, model_name, input_shape)
        model = inc_net.InceptionV3()

    explainable_image = img_processed.copy()
    prediction, prediction_score, prediction_type = get_predictions(model_name, model, img_processed.copy())
    
    explanation_custom = get_explanations('CIE_INSPIRATION', explainable_image, model, model_name)
    # mask_normed = (explanation_custom - explanation_custom.mean())/(explanation_custom.std())
    
    explanation_custom_2 = get_explanations('FEATURE_EXTRACTION_LAST_LAYER', explainable_image, model, model_name)
    
    return json.dumps(
        {'prediction': prediction, 'score': str(prediction_score),
         'cie_inspiriation': explanation_custom.tolist(),'explanation_custom_2':explanation_custom_2.tolist()})

@app.post("/benchmarks")
async def benchmarks(request: Request, file: Union[UploadFile, None] = None):
    form = await request.form()
    contents = form["upload_file"].file
    data = contents.read()
    model_name = form['model_name']
    if model_name == 'VGG16':
        input_shape = (224, 224)
        img_processed = get_img_arr(data, model_name, input_shape)
        model = VGG16()
    else:
        input_shape = (229, 229)
        img_processed = get_img_arr(data, model_name, input_shape)
        model = inc_net.InceptionV3()
    explainable_image = img_processed.copy()
    prediction, prediction_score, prediction_type = get_predictions(model_name, model, img_processed.copy())
    explanation_lime = get_explanations('LIME', explainable_image, model, model_name)
    return json.dumps({'prediction': prediction, 'score': str(prediction_score), 'heatmap_lime': explanation_lime.tolist()})

if __name__ == "__main__":
    uvicorn.run(app, debug=True)
