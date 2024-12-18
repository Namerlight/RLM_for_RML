# in this file we have functions like "run_gpt2" or "run_llama3" or sth
# each of these functions will take an input text, then do the prediction, then return the output string
# we're doing it this way because we might have to individually pre/post-process each model


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


list_of_models = {
    "gpt2": "gpt2",
    "gemma2_2B": "google/gemma-2-2b-it",
    "llama3.2_1B": "meta-llama/Llama-3.2-1B",
    "llama3.2_3B": "meta-llama/Llama-3.2-3B",
    "llama3_8B": "meta-llama/Meta-Llama-3-8B-Instruct",
}

types_of_prompts = [
    "Role",
    "Style",
    "Emotion",
    "Rephrase",
    "Reread",
    "Self-Ask",
    "Zero-Shot-CoT",
    "Step-Back",
    "Least-to-Most",
    "Self-Calibration",
]


class answer_generator:
    def __init__(self, model):

        models_list = list_of_models

        if models_list.get(model) is None:
            available_models = "\n".join([key for key in models_list.keys()])
            raise Exception(f'Model not found. The available models are:\n{available_models}')

        hf_model_name = models_list.get(model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=hf_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=hf_model_name, device_map=(0 if torch.cuda.is_available() else -1), torch_dtype=torch.bfloat16,
        )

    def build_prompt(self, input_text: str, type_of_prompt):

        if type_of_prompt not in types_of_prompts:
            available_prompts = "\n".join(types_of_prompts)
            raise Exception(f'Prompt type not found. The available prompting methods are:\n{available_prompts}')

        if type_of_prompt == "Role":
            prompt = ("You are a trivia expert with a lot of factual knowledge. Answer the following question succintly. "
                      ""
                      f"Question: {input_text} "
                      "Answer: ")
        elif type_of_prompt == "Style":
            prompt = ("Answer the following question in a clear and factual style, without any unnecessary embellishments. "
                      ""
                      f"Question: {input_text} "
                      "Answer: ")
        elif type_of_prompt == "Emotion":
            prompt = ("I need the answer to the following question. It's really important for my health and wellbeing. If you give the wrong answer, I could be in danger, so please answer correctly. "
                      ""
                      f"Question: {input_text} "
                      "Answer: ")
        elif type_of_prompt == "Rephrase":
            prompt = ("Please rephrase the following question. Make the question clear and remove any unnecessary information. "
                      ""
                      f"Question: {input_text} "
                      "Rephrased Question: ")
            prompt = self.intermediate_inference(prompt)
        elif type_of_prompt == "Reread":
            prompt= (f"Question: {input_text} "
                     f"Answer: {self.intermediate_inference(input_text)} "
                     ""
                     f"Please re-read the question and answer again: {input_text}"
                     "Answer: ")
        elif type_of_prompt == "Self-Ask":
            prompt = (f"Consider this question: {input_text} "
                      "In order to answer this question, do you need to ask any questions in order to answer this question?")
            intermed_questions = self.intermediate_inference(prompt)
            intermed_answers = self.intermediate_inference(intermed_questions)
            prompt = (f"Consider this question: {input_text} "
                      f"Here is some extra information to help you answer: {intermed_answers} "
                      "Now, answer the question above. "
                      ""
                      "Answer: ")
        elif type_of_prompt == "Zero-Shot-CoT":
            prompt = (f"Consider this question: {input_text} "
                      "Is this question about a specific topic or individual?")
            intermed_response = self.intermediate_inference(prompt)
            prompt = (f"User: Consider this question: {input_text} "
                      "Is this question about a specific topic or individual?"
                      "Assistant: This question is about the following things:"
                      f"{intermed_response}"
                      "User: For each thing listed above, list everything you know about it."
                      )
            intermed_response = self.intermediate_inference(prompt)
            prompt = (f"User: Consider this question: {input_text} "
                      "Is this question about a specific topic or individual?"
                      "Assistant: This question is about the following things:"
                      f"{intermed_response}"
                      "User: For each thing listed above, list everything you know about it."
                      f"Assistant: {intermed_response}"
                      "User: Based on the information above, please answer the original question."
                      f"Question: {input_text} "
                      "Answer: ")
        elif type_of_prompt == "Step-Back":
            prompt = (f"Question: {input_text} "
                      "Answer: ")
            intermed_response = self.intermediate_inference(prompt)
            prompt = (f"Question: {input_text} "
                      f"Answer: {intermed_response} "
                      "User: Please take a step back and consider your response again.")
        elif type_of_prompt == "Least-to-Most":
            prompt = (f"Question: {input_text}. "
                      f"Please give the answer as a list of responses, from least likely to most likely"
                      "Answers: ")
            intermed_response = self.intermediate_inference(prompt)
            prompt = (f"{intermed_response}"
                      f"Please return the last item in the above list as a sentence. Do not add anything before or after the sentence.")
        elif type_of_prompt == "Self-Calibration":
            prompt = (f"Question: {input_text} "
                      "Answer: "
                      ""
                      "Is the answer to the question above correct? If not, please fix the mistakes. If it is correct, simply respond with the same answer."
                      "Corrected Answer: ")

        return prompt

    def intermediate_inference(self, input_text: str) -> str:

        input_ids = self.tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**input_ids, max_new_tokens=128, pad_token_id=self.tokenizer.eos_token_id)
        answer = self.tokenizer.decode(outputs[0])

        for token in set(self.tokenizer.all_special_tokens):
            answer = answer.replace(token, "").strip()

        return answer.replace(input_text, "").strip()


    def run_HF_models(self, input_text: str, prompt) -> [str]:
        """
        Processes input text in the given model.

        Args:
            input_text: text to input

        Returns:
            output text from gemma2.
        """

        if prompt:
            prompt = self.build_prompt(input_text=input_text, type_of_prompt=prompt)
        else:
            prompt = input_text

        input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**input_ids, max_new_tokens=256, pad_token_id=self.tokenizer.eos_token_id)
        answer = self.tokenizer.decode(outputs[0])

        for token in set(self.tokenizer.all_special_tokens):
            answer = answer.replace(token, "").strip()

        answer = answer.replace(prompt, "").strip()

        return prompt, answer


def get_list_of_models():
    return list_of_models


def get_list_of_prompts():
    return types_of_prompts
