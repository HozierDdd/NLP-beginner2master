from openai import OpenAI
import openai
from config import OPENAI_API_KEY
from file_loader import FileLoader
from typing import Optional



class TrainedModel():
    def __init__(self, api_key, fine_tune_id):
        self.api_key = api_key
        openai.api_key = self.api_key  # Set OpenAI API key
        self.fine_tune_id = fine_tune_id
        # self.fine_tune_job = openai.FineTune.retrieve(id=self.fine_tune_id)
        self.fine_tune_job = openai.fine_tuning.jobs.retrieve(fine_tuning_job_id=self.fine_tune_id)

    def check_status(self):
        if self.fine_tune_job.status == 'succeeded':
            fine_tuned_model = self.fine_tune_job.fine_tuned_model
            print(f"Fine-tuned model ID: {fine_tuned_model}")
            return True
        else:
            print("Fine-tuning job is not completed yet.")
            return False

    def use_trained_model(self):
        if self.check_status():
            try:
                client = OpenAI(api_key=self.api_key)
                fine_tuned_model = self.fine_tune_job.fine_tuned_model
                completion = client.chat.completions.create(
                    model=fine_tuned_model,  # fine-tuned-model-id
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"}
                    ]
                )
                print(completion.choices[0].message)
            except Exception as e:
                print(f"Error occurred: {e}")


if __name__ == "__main__":
    api_key = OPENAI_API_KEY
    fine_tune_id = "ftjob-A2CrJpWVbYtnLju1QHNmWZQQ"  # Replace with fine-tune job ID
    trained_model = TrainedModel(api_key, fine_tune_id)
    trained_model.use_trained_model()