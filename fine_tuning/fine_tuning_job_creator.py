import openai
import config


class FineTuningJobCreator:
    def __init__(self, api_key, model):
        openai.api_key = api_key
        self.model = model

    def create_job(self):
        response = openai.FineTune.create(
            training_file="file-abc123",
            model=self.model
        )
        return response

    def create_job_legacy(self):
        client = OpenAI()
        client.fine_tuning.jobs.create(
            training_file="file-abc123",
            model=self.model
        )
        # List 10 fine_tuning jobs
        client.fine_tuning.jobs.list(limit=10)

        # Retrieve the state of a fine-tune
        client.fine_tuning.jobs.retrieve("ftjob-abc123")

        # Cancel a job
        client.fine_tuning.jobs.cancel("ftjob-abc123")

        # List up to 10 events from a fine_tuning job
        client.fine_tuning.jobs.list_events(fine_tuning_job_id="ftjob-abc123", limit=10)

        # Delete a fine-tuned model (must be an owner of the org the model was created in)
        client.models.delete("ft:gpt-3.5-turbo:acemeco:suffix:abc123")


if __name__ == "__main__":
    # Example usage:
    api_key = config.OPENAI_API_KEY
    model = 'gpt-3.5-turbo'
    job_creator = FineTuningJobCreator(api_key, model)
    response = job_creator.create_job()
    print(response)