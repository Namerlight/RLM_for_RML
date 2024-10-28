# in this file we have functions like "run_gpt2" or "run_llama3" or sth
# each of these functions will take an input text, then do the prediction, then return the output string
# we're doing it this way because we might have to individually pre/post-process each model

import torch
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_gpt2(input_text: str, seed = None) -> [str]:
    """
    Processes input text in gpt2.

    Args:
        input_text: text to input
        seed: seed for random number generator

    Returns:
        output text from gpt2. This is a list of strings (unless we change num_return_sequences later)
    """

    generator = pipeline('text-generation', model='gpt2')
    if seed: set_seed(42)
    output = generator(input_text, max_length=30, num_return_sequences=5)
    generated_text = [gendictout['generated_text'] for gendictout in output]

    return generated_text


def run_gemma2_2B(input_text: str, seed = None) -> [str]:
    """
    Processes input text in gemma2.
    Args:
        input_text: text to input
        seed: seed for random number generator

    Returns:
        output text from gemma2.
    """

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it", device_map="auto", torch_dtype=torch.bfloat16,
    )

    input_text = input_text
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids, max_new_tokens=32)

    return tokenizer.decode(outputs[0])


models_list = {
    "gpt2": run_gpt2,
    "gemma2_2B": run_gemma2_2B
}