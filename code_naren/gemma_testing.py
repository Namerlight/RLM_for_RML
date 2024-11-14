import torch
from transformers import pipeline
import pandas as pd

pipe = pipeline(
        "text-generation",
        model="google/gemma-2-9b",
        device="cuda",  # replace with "mps" to run on a Mac device
    )
outputs = pipe("You are a knowledgable historian, answer this question with the considerable expertise at your disposal. What happens when you walk into a wardrobe?", max_new_tokens=256)

response = outputs[0]["generated_text"]
print(response)