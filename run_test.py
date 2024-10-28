import argparse
import scripts

def main(model, data):
    """
    Run a specific model through a full test dataset.

    Args:
        model: which model to run. This is actually passed as a function name (e.g. "run_gpt2")
        data: directions to a dataset or a function that loads a dataset

    Returns:
        test scores, accuracy, some sample results, etc.
        Can just output to a file in results/
    """

    model_to_run = scripts.models_list.get(model)

    if model_to_run is None:
        available_models = "\n".join([key for key in scripts.models_list.keys()])
        raise Exception(f'Model not found. The available models are:\n{available_models}')

    print(model_to_run(input_text="Hello there, General"))

    # if model == "gpt2":

    # else
    # function_to_call = some other model

    # do something here if necessary
    # loop through dataset

    # if you need to process the data in any way
    # do it in a function inside the datasets/ folder and call that here

    # for loop in data:
    #     function_to_call(loop)

    # output results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ### arguments for which model to run and which dataset to run

    # main(parsed arguments)
    main(model="gemma2_2B", data="")