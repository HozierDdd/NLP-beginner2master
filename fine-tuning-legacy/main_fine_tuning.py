from openai import OpenAI
from check_dataset.check_dataset_format_for_3p5 import CheckDatasetFormatFor3P5


"""STEP 1: check dataset format"""
check_dataset = CheckDatasetFormatFor3P5()
check_dataset.error_check()
"""STEP 2: craft prompts"""

"""STEP 3: upload dataset to OpenAI"""

"""STEP 4: create fine-tuning-legacy job"""

"""STEP 5: use trained model to generate responses"""

"""STEP 6: create and use trained model"""
