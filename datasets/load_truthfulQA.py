import os
import numpy as np
import pandas as pd


pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)


def convert_to_labelwise(questions: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the TruthfulQA dataset in dataframe format into a label-wise format.
    Each row contains a single answer (Answers columns) and a label (IsAnswer column) of correct or incorrect.

    Args:
        questions: dataframe for questions. This will only work with the TruthfulQA dataset format and columns.

    Returns:
        dataframe with row-wise labels format
    """

    questions['Correct Answers'] = questions.apply(lambda row: [row['Best Answer']] + row['Correct Answers'], axis=1)
    questions.drop('Best Answer', axis=1, inplace=True)

    correct_responses = questions[['Type', 'Category', 'Question', 'Correct Answers']].explode('Correct Answers').drop_duplicates()
    correct_responses["IsAnswer"] = "Correct"
    correct_responses.rename(columns={'Correct Answers': 'Answers'}, inplace=True)

    wrong_responses = questions[['Type', 'Category', 'Question', 'Incorrect Answers']].explode('Incorrect Answers').drop_duplicates()
    wrong_responses["IsAnswer"] = "Incorrect"
    wrong_responses.rename(columns={'Incorrect Answers': 'Answers'}, inplace=True)

    questions_labelwise = pd.concat([correct_responses, wrong_responses], ignore_index=True)

    original_order = pd.unique(questions_labelwise['Question'])
    questions_labelwise['order'] = questions_labelwise['Question'].apply(lambda x: np.where(original_order == x)[0][0])
    questions_labelwise = questions_labelwise.sort_values(['order', 'Question']).drop('order', axis=1).reset_index(drop=True)

    return questions_labelwise


def fetch_TruthfulQA(dataset_path: str = None, labelwise: bool = True) -> pd.DataFrame:
    """
    Just fetches the TruthfulQA dataset from the given path (defaults to ../data/TruthfulQA/TruthfulQA.csv).
    If labelwise is True, calls convert_to_labelwise and returns the dataset in that format.

    Args:
        dataset_path: path to TruthfulQA dataset. Defaults to ../data/TruthfulQA/TruthfulQA.csv
        labelwise: whether to return label-wise or not. Defaults to True.

    Returns:
        dataframe for TruthfulQA dataset.
    """

    if not dataset_path:
        dataset_path = os.path.join("..", "data", "TruthfulQA", "TruthfulQA.csv")

    questions = pd.read_csv(dataset_path)
    questions.dropna(axis=1, how='all', inplace=True)

    questions["Correct Answers"] = questions["Correct Answers"].str.split("; ")
    questions["Incorrect Answers"] = questions["Incorrect Answers"].str.split("; ")

    if labelwise:
        questions = convert_to_labelwise(questions)

    return questions


# In case you want to test these outputs.
if __name__ == "__main__":
    questions_df = fetch_TruthfulQA()
    updated_questions_df = convert_to_labelwise(questions=questions_df)