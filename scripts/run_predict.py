# in this file we have functions like "run_gpt2" or "run_llama3" or sth
# each of these functions will take an input text, then do the prediction, then return the output string
# we're doing it this way because we might have to individually pre/post-process each model

from transformers import pipeline, set_seed


def run_gpt2(input_text: str) -> str:
    """
    Processes input text in gpt2.

    Args:
        input_text: text to input

    Returns: output text from gpt2. This may be as a list or as a string
    """
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

    return output

