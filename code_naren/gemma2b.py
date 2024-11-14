import torch
from transformers import pipeline
import pandas as pd


prompts = ["You are a knowledgeable historian. Answer the following question with accuracy and clarity one sentence answers, no commas allowed.", 
           "Provide a concise  factual response to this question  with a focus on clarity and simplicityone sentence answers  no commas allowed.",
           "Answer the question with an encouraging and friendly tone one sentence answers  no commas allowed.",
           "Rephrase this question in simpler terms and then provide an answer one sentence answers  no commas allowed.",
           "Read the question carefully  then read it again to ensure clarity. After understanding fully  provide a thoughtful answer one sentence answers  no commas allowed.",
           "Break down the question by asking yourself clarifying sub-questions. After you answer these  summarize the answer to the main question one sentence answers  no commas allowed.",
           "Think step-by-step to reach the answer to the question. Take your time  here's the question one sentence answers  no commas allowed.",
           "Answer the question  then pause and evaluate if the answer is complete. If anything is missing  add it to improve clarity one sentence answers  no commas allowed.",
           "Answer the question by starting with simple details  then build up to more complex information one sentence answers  no commas allowed.",
           "Give an answer to the question  then provide a confidence level (high  medium  or low) based on how certain you are one sentence answers  no commas allowed.",
           "Answer the question  then review your response. If you notice any improvements that could clarify or add value to your answer  revise it accordingly one sentence answers  no commas allowed."]


opfile_path = "D://Github//RLM_for_RML//code_naren//collected_data.csv"

def random_sample():

    file_path = 'D://Github//RLM_for_RML//code_naren//TruthfulQA.csv' 
    data = pd.read_csv(file_path)
    sentences = data['Question']
    sampled_sentences = sentences.sample(n=20, random_state=32)  # random_state for reproducibility
    questions = sampled_sentences.tolist()
    print(questions)
    return questions

def prompt_the_model(prompt, question):
    text = prompt+" "+question
    print(text)
    pipe = pipeline(
        "text-generation",
        model="google/gemma-2-2b",
        device="cuda",  # replace with "mps" to run on a Mac device
    )
    outputs = pipe(text, max_new_tokens=256)
    return outputs


questions = random_sample()
for i in range(0,10):
    for j in range(0, len(questions)):
        output = prompt_the_model(prompts[i], questions[j])
        response = output[0]["generated_text"]
        print(response)
        with open(opfile_path, 'a') as file:
            file.write(str(i)+","+str(j)+","+response+","+"gemma 2"+"\n")


