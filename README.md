# XAI TECHNIQUES - LIME, CIE INSPIRATIONS, FEATURE EXPLAINER TECHNIQUES FOR VGG16 and INCEPTION V3 PRE TRAINED MODELS

## Project setup
  - conda create -y python=3.9 --name xai
  - conda activate xai
  - pip install -r requirements.txt
  - conda activate xai
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