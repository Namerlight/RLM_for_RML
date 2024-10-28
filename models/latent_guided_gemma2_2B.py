import torch
from transformers import Gemma2Model, Gemma2Config
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers.models.gemma.convert_gemma_weights_to_hf import gemma_2b_config


def infer_gemma2_2B(input_text):

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it", device_map="auto", torch_dtype=torch.bfloat16,
        output_hidden_states=True, return_dict_in_generate=True
    )

    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=16)
    layers = outputs.get("hidden_states")

    # print(len(layers))
    # for ly in layers:
    #     print(len(ly))
    #     print(ly)

    last_ly = layers[-1]
    print("Length of last layer", len(last_ly))
    print("Values in last layer", last_ly[0])

    # print("Final output length:", len(outputs["sequences"][0]))
    return tokenizer.decode(outputs["sequences"][0]).replace(input_text, "")


op = infer_gemma2_2B("What happens to you if you eat watermelon seeds?")
print(op)