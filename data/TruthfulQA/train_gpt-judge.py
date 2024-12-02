import os
from openai import OpenAI
from configs import org, api_key
import json

client = OpenAI(
    organization=org,
    api_key=api_key
)

# file_name = client.files.create(
#   file=open(os.path.join("data", "chat_finetune_truth.jsonl"), "rb"),
#   purpose="fine-tune"
# )
#
# print(file_name)

client.fine_tuning.jobs.create(
  training_file="file-G73vU8Y7DmZuF5GoqD6Mph8f",
  model="gpt-3.5-turbo-0125"
)


