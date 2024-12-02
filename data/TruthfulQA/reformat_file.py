import os
import json

with (open(os.path.join("data", "finetune_truth.jsonl"), "r", encoding="utf-8") as infile,
      open(os.path.join("data", "chat_finetune_truth.jsonl"), "w", encoding="utf-8") as outfile):
    for line in infile:
        data = json.loads(line)

        prompt = data["prompt"]
        completion = data["completion"].strip()

        # question = prompt.split("\nA:")[0].strip()
        # answer = prompt.split("\nA:")[1].split("\nHelpful:")[0].strip()

        chat_format = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]
        }

        json.dump(chat_format, outfile)
        outfile.write("\n")