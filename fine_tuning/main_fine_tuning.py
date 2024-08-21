# from openai import OpenAI
import openai
from check_dataset.check_dataset_format_for_3p5 import CheckDatasetFormatFor3P5
from utls import Utils
from config import OPENAI_API_KEY
from file_loader import FileLoader
from fine_tuning_job_creator import FineTuningJobCreator
from apply_trained_model import TrainedModel
"""STEP 0: initialize environment"""
filepath = "dataset/prompt_3.5_train.jsonl"
api_key = OPENAI_API_KEY
# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",  # or another model from the updated API
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What are the benefits of eating healthy foods?"}
#     ]
# )
# print(response.choices[0].message['content'])
"""STEP 1: check dataset format"""
check_dataset = CheckDatasetFormatFor3P5(filepath=filepath)
check_dataset.error_check()
"""STEP 2: craft prompts from the dataset"""
# for item in check_dataset.dataset:
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=item['messages']
#     )
#     print(response.choices[0].message['content'].strip())

"""STEP 3: upload dataset to OpenAI"""
file_loader = FileLoader(directory_path=filepath, api_key=OPENAI_API_KEY)
file_id = file_loader.load_files()  # This should be a remote file ID, not a local file path. You can find this ID
# from the response from the openai.File.create
"""STEP 4: create fine_tuning job"""
fine_tuning_job_creator = FineTuningJobCreator(api_key=OPENAI_API_KEY, model='gpt-3.5-turbo', file_name_remote=file_id)
response = fine_tuning_job_creator.create_job()
print(response)
"""STEP 5: use trained model to generate responses"""
trained_model = TrainedModel(api_key=OPENAI_API_KEY, fine_tune_id=response.id)
trained_model.use_trained_model()
"""STEP 6: create and use trained model"""
