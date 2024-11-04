import os
import scripts
import argparse
import pandas as pd

from scripts.run_predict import answer_generator
from scripts.run_eval import eval_results


def main(model, data, save = None):
    """
    Run a specific model through a full test dataset.

    Args:
        model: which model to run. This is actually passed as a function name (e.g. "run_gpt2")
        data: directions to a dataset or a function that loads a dataset
        save: path to save file for dataset with answers listed.

    Returns:
        test scores, accuracy, some sample results, etc.
        Can just output to a file in results/
    """

    dataset = pd.read_csv(data)

    ag = answer_generator(model=model)

    dataset[f'{model}_answer'] = dataset['Question'].apply(lambda q: ag.run_HF_models(input_text=q))

    if save:
        dataset.to_csv(path_or_buf=save, index=False)


def eval(model, data, save = None):
    """
    Run a specific model through a full evaluation step.

    Args:
        model: which model to use to evaluate
        data: directions to a dataset or a function that loads a dataset
        save: path to save file for dataset with answers listed.

    Returns:
        test scores, accuracy, some sample results, etc.
        Can just output to a file in results/
    """


    # eval_results(data=data, model=model, save=save)

    answersset = pd.read_csv(save)

    total_rows = len(answersset)
    yes_count = answersset['Correct'].str.lower().eq("yes").sum()
    accuracy = (yes_count / total_rows) * 100 if total_rows > 0 else 0

    print(f"Model: {model}\nAccuracy: {accuracy}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ### arguments for which model to run and which dataset to run
    # main(parsed arguments)

    data_path = os.path.join("data", "TruthfulQA")
    # main(model="gemma2_2B", data=os.path.join(data_path, "TruthfulQA.csv"), save=os.path.join(data_path, "TruthfulQA_answers_gemma2.csv"))
    # main(model="llama3_3B", data=os.path.join(data_path, "TruthfulQA.csv"), save=os.path.join(data_path, "TruthfulQA_answers_llama3.csv"))

    eval(model="gptjudge", data=os.path.join(data_path, "TruthfulQA_answers_gemma2.csv"),
         save=os.path.join(data_path, "TruthfulQA_answers_gemma2_labelled.csv"))