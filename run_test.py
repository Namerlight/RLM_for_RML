import argparse
from scripts import run_gpt2


def main(model, data):
    """
    Run a specific model through a full test dataset.

    Args:
        model: which model to run. This is actually passed as a function name (e.g. "run_gpt2")
        data: directions to a dataset or a function that loads a dataset

    Returns: test scores, accuracy, some sample results, etc.
    Can just output to a file in results/
    """

    # if model == "gpt2":
    function_to_call = run_gpt2
    # else
    # function_to_call = some other model

    # do something here if necessary
    # loop through dataset

    # if you need to process the data in any way
    # do it in a function inside the datasets/ folder and call that here

    for loop in data:
        function_to_call(loop)

    # output results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ### arguments for which model to run and which dataset to run

    # main(parsed arguments)