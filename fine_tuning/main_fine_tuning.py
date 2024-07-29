# from openai import OpenAI
import openai
from check_dataset.check_dataset_format_for_3p5 import CheckDatasetFormatFor3P5
from utls import Utils
from config import OPENAI_API_KEY

"""STEP 0: initialize environment"""
filepath = "dataset/prompt_3.5_train.jsonl"
openai.api_key = OPENAI_API_KEY
# Example of making a request
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="What are the benefits of eating healthy foods?",
    max_tokens=50
)

print(response.choices[0].text.strip())
"""STEP 1: check dataset format"""
check_dataset = CheckDatasetFormatFor3P5(filepath=filepath)
check_dataset.error_check()
"""STEP 2: craft prompts"""

"""STEP 3: upload dataset to OpenAI"""
utils = Utils(dataset=check_dataset.dataset, filepath=filepath)
utils.upload()
"""STEP 4: create fine_tuning job"""
utils.create_job()
"""STEP 5: use trained model to generate responses"""
utils.use_trained_model()
"""STEP 6: create and use trained model"""