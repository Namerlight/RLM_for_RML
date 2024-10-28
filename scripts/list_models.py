from run_predict import models_list

available_models = "\n".join([key for key in models_list.models_list.keys()])

print("The models implemented are:\n", available_models)
