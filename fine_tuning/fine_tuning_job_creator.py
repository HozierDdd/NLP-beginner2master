import openai
from openai import OpenAI
import config

class FineTuningJobCreator:
    def __init__(self, api_key, model, file_name_remote):
        # openai.api_key = api_key
        self.api_key = api_key
        self.model = model
        self.file_name_remote = file_name_remote  # This should be a remote file ID, not a local file path. You can find this ID from the response from the File.create() API call.
        self.model = model
        self.file_name_remote = file_name_remote  # This should be a remote file ID, not a local file path. You can find this ID from the response from the File.create() API call.

    def create_job(self):
        client = OpenAI(api_key=self.api_key)
        try:
            response = client.fine_tuning.jobs.create(
                training_file=self.file_name_remote,
                model=self.model
            )
            return response
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def create_job_legacy(self):
        client = openai
        client.FineTune.create(
            training_file=self.file_name_remote,
            model=self.model
        )
        # List 10 fine_tuning jobs
        client.FineTune.jobs.list(limit=10)

        # Retrieve the state of a fine-tune
        client.FineTune.jobs.retrieve("ftjob-abc123")

        # Cancel a job
        client.FineTune.jobs.cancel("ftjob-abc123")

        # List up to 10 events from a fine_tuning job
        client.FineTune.jobs.list_events(fine_tuning_job_id="ftjob-abc123", limit=10)

        # Delete a fine-tuned model (must be an owner of the org the model was created in)
        client.Model.delete("ft:gpt-3.5-turbo:acemeco:suffix:abc123")


if __name__ == "__main__":
    # Example usage:
    api_key = config.OPENAI_API_KEY
    model = 'gpt-3.5-turbo'  # 'gpt-3.5-turbo' or 'davinci'
    # directory_path = "dataset/prompt_3.5_train.jsonl"
    # file_loader = FileLoader(directory_path, api_key)
    # file_id = file_loader.load_files()
    job_creator = FineTuningJobCreator(api_key, model, file_name_remote="file-gj75r35DD1WrjH2OCC1QMi5i")
    response = job_creator.create_job()
    print(response)