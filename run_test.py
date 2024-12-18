import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch, accelerate

from scripts.run_predict import answer_generator
from scripts.run_truthfulqa_eval import truthfulqa_eval_results
from scripts.run_simpleqa_eval import simpleqa_eval_results

tqdm.pandas()


def main(model, data, prompt=None, num_examples=None, save=None, sample_seed=1):
    """
    Run a specific model through a full test dataset.

    Args:
        model: which model to run. This is actually passed as a function name (e.g. "run_gpt2")
        data: directions to a dataset or a function that loads a dataset
        prompt: which type of prompting to use.
        save: path to save file for dataset with answers listed.
        num_examples: number of examples from the dataset to run. Defaults to None to run full dataset
        sample_seed: seed for sampling examples at random. Unused if num_examples isn't used.

    Returns:
        test scores, accuracy, some sample results, etc.
        Can just output to a file in results/
    """

    dataset = pd.read_csv(data)

    if num_examples: dataset = dataset.sample(n=num_examples, random_state=sample_seed)

    questions_col = "problem" if "simple_qa" in data else "Question"

    ag = answer_generator(model=model)

    print(f"\nComputing answers for {model} with {data} and {prompt}")
    dataset['output'] = dataset[questions_col].progress_apply(lambda q: ag.run_HF_models(input_text=q, prompt=prompt))
    dataset['prompt'], dataset['pred_answer'] = zip(*dataset['output'])
    dataset.drop(columns=['output'], inplace=True)

    if save:
        dataset.to_csv(path_or_buf=save, index=False)


def eval(model, data, prompt, save, metrics_only = False):
    """
    Run a specific model through a full evaluation step.

    Args:
        model: which model to use to evaluate
        data: directions to a dataset or a function that loads a dataset
        prompt: the prompt that was originally passed into the model
        save: path to save file for dataset with answers listed.
        metrics_only: if True, only evaluate metrics, otherwise evaluate results then generate metrics

    Returns:
        test scores, accuracy, some sample results, etc.
        Can just output to a file in results/
    """

    if not metrics_only:

        # gptjudge is implemented, but performs poorly, so I've hardcoded simplejudge
        if model == "gptjudge":
            truthfulqa_eval_results(data=data, model=model, save=save)
        elif model == "simplejudge":
            simpleqa_eval_results(data=data, save=save)

    answers_set = pd.read_csv(save)

    total_rows = len(answers_set)
    yes_count = answers_set['Correct'].str.lower().eq("yes").sum()
    accuracy = (yes_count / total_rows) * 100 if total_rows > 0 else 0

    op = f"Model: {model} | Data: {data} | Prompt: {prompt} | Accuracy: {accuracy}%\n"

    # with open(os.path.join("results", "prompt_eng_results.txt"), 'a') as file:
    #     print(op, file=file)

    with open(os.path.join("results", "prompt_eng_results_2.txt"), 'a') as file:
        print(op, file=file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    model_to_use = "gemma2_2B"
    datasets = ["truthfulqa", "simpleqa"]

    for qs_data in datasets:

        # prompts_list = ["Rephrase", "Reread", "Self-Ask", "Zero-Shot-CoT"]
        prompts_list_2 = ["Self-Calibration"] # "Step-Back", "Least-to-Most", ]

        d_set, s_set = {
            "truthfulqa": os.path.join("data", "TruthfulQA", "TruthfulQA.csv"),
            "simpleqa": os.path.join("data", "simple_evals", "simple_qa_test_set.csv"),
        }, {
            "truthfulqa": os.path.join("data", "generated_answers", f"TruthfulQA_answers_{model_to_use}.csv"),
            "simpleqa": os.path.join("data", "generated_answers", f"simple_qa_test_set_answers_{model_to_use}.csv"),
        }

        data_path = d_set.get(qs_data)

        for pr in prompts_list_2:

            main(model=model_to_use, data=d_set.get(qs_data), save=s_set.get(qs_data), num_examples=100, prompt=pr)
            eval(model="simplejudge", data=s_set.get(qs_data), save=s_set.get(qs_data).replace(".csv", "_labelled.csv"), prompt=pr)

        torch.cuda.empty_cache()
