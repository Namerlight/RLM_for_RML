# RLM_for_RML (name tbd, okay)

Robust Language Modelling for Truthful QA

## setup

Developed on Python 3.9 &nbsp; <sub><sup>(note - this was 3.11 earlier)</sup></sub>

Run the following:

```bash
cd data
git clone https://github.com/sylinrl/TruthfulQA.git
cd ../
pip install -r requirements.txt
```

## To-Do

* [X] Set up dataset (truthfulQA) and preprocess it if needed
* [ ] Set up testing pipeline in run_test.py
  * [X] **you don't actually have to set up from scratch**, just do the local installation from instructions here (https://github.com/sylinrl/TruthfulQA), then add a function in datasets to call the appropriate functions there
  * [ ] they only let you use huggingface models so you will have to do some customizing so we can run our custom models. Maybe look at their truthfulqa/evaluate.py file and get code from there
* [X] implement function to extract embeddings from an LM
* [ ] then implement function to get the difference between truthful and false embeddings
* [ ] then add the difference to the LM at inference time
* [ ] coming soon