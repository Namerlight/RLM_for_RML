from scripts import get_list_of_models

available_models = "\n".join([key for key in get_list_of_models().keys()])

print("The models implemented are:", available_models, sep="\n")
