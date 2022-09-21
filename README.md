# XAI TECHNIQUES - LIME, CIE INSPIRATIONS, FEATURE EXPLAINER TECHNIQUES FOR VGG16 PRE TRAINED MODELS


## About the project

The project consists of two pages. The first page displays explanations of the VGG16 neural network classification result using LIME explainers, which is used in benchmarking. The second page displays explanations of custom developed explainers that are influenced by counterfactual explanations, as well as maps the pixel weights of the second last layer of the VGG16 model to provide insights into the most contributing pixels for categorization.


### Page 1
![alt text](./assets/Page-1.png)
LIME Explanation


### Page 2
![alt text](./assets/Page-2.png)
Exploring the pixel weights in the second last VGG16 convolutional layer.

![alt text](./assets/Page-3.png)
Counterfactual explanations sighting minor changes in which pixel cause the most disruption in the prediction.


## Project setup
  - conda create -y python=3.9 --name xai
  - conda activate xai
  - pip install -r requirements.txt
  - Rename config-dist.py to config.py and update as per your local machine configurations
  
## To Run API
  - uvicorn api:app --reload
  - Explainer endpoint - http://127.0.0.1:8000/explain
  - Benchmark endpoint - http://127.0.0.1:8000/benchmark
  
## To Run Streamlit / Dashboard to view explanations
  - streamlit run actionlearning.py
  - Dashboard endpoint - http://localhost:8501/

## PAGES
### PAGE 1 - Dashboard to establish benchmarks for explainer using pre-existing packages such as LIME
  - Upload the file
  - Press submit button to view explanations

### PAGE 1 - Dashboard to view explanations inspired from CIE and Feature Analysis
  - Upload the file
  - Press submit button to view explanations