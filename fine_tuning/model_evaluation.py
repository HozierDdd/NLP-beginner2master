import openai
from file_loader import FileLoader
from typing import Optional
from config import OPENAI_API_KEY


class ModelEvaluation():

    def __init__(self, api_key, fine_tune_id, eval_file_path: Optional[str] = "dataset/prompt_3.5_eval.jsonl"):
        self.api_key = api_key
        openai.api_key = self.api_key  # Set OpenAI API key
        self.fine_tune_id = fine_tune_id
        self.file_loader = FileLoader(directory_path=eval_file_path, api_key=self.api_key)
        # self.fine_tune_job = openai.FineTune.retrieve(id=self.fine_tune_id)
        self.fine_tune_job = openai.fine_tuning.jobs.retrieve(fine_tuning_job_id=self.fine_tune_id)

    def evaluate_model(self, model_id):
        file_id = self.file_loader.load_files()
        correct_predictions = 0
        total_predictions = len(validation_data)

        for data in validation_data:
            prompt = data['prompt']
            expected_completion = data['expected_completion']

            # Generate prediction using the fine-tuned model
            response = openai.Completion.create(
                model=model_id,
                prompt=prompt,
                max_tokens=50
            )
            predicted_completion = response.choices[0].text.strip()

            # Simple accuracy check (exact match)
            if predicted_completion == expected_completion:
                correct_predictions += 1

            # Print each prediction for review
            print(f"Prompt: {prompt}")
            print(f"Expected: {expected_completion}")
            print(f"Predicted: {predicted_completion}")
            print()

        # Calculate accuracy
        accuracy = correct_predictions / total_predictions
        return accuracy


if __name__ == "__main__":
    model_eval = ModelEvaluation(api_key=OPENAI_API_KEY, fine_tune_id="ftjob-A2CrJpWVbYtnLju1QHNmWZQQ")
    model_eval.evaluate_model("ft:gpt-3.5-turbo-0125:personal::9qQShW9q")