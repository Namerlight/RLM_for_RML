# in this file we have functions like "run_gpt2" or "run_llama3" or sth
# each of these functions will take an input text, then do the prediction, then return the output string
# we're doing it this way because we might have to individually pre/post-process each model


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



class answer_generator:
    def __init__(self, model):

        models_list = {
            "gpt2": "gpt2",
            "gemma2_2B": "google/gemma-2-2b-it",
            "llama3_1B": "meta-llama/Llama-3.2-1B",
            "llama3_3B": "meta-llama/Llama-3.2-3B"
        }

        if models_list.get(model) is None:
            available_models = "\n".join([key for key in models_list.keys()])
            raise Exception(f'Model not found. The available models are:\n{available_models}')

        hf_model_name = models_list.get(model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=hf_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=hf_model_name, device_map="auto", torch_dtype=torch.bfloat16,
        )


    def run_HF_models(self, input_text: str, seed = None) -> [str]:
        """
        Processes input text in gemma2's 2B model.
        Args:
            input_text: text to input
            seed: seed for random number generator

        Returns:
            output text from gemma2.
        """

        input_text = input_text
        input_ids = self.tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**input_ids, max_new_tokens=32)
        answer = self.tokenizer.decode(outputs[0])

        special_tokens = set(self.tokenizer.all_special_tokens)

        for token in special_tokens:
            answer = answer.replace(token, "").strip()

        return answer
