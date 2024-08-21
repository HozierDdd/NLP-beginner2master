import openai
import config


class FileLoader():
    def __init__(self, directory_path, api_key):
        self.directory_path = directory_path
        self.api_key = api_key

    def load_files(self):
        # Set your API key
        openai.api_key = self.api_key

        # Upload the training file
        response = openai.File.create(
            file=open(self.directory_path, "rb"),
            purpose="fine-tune"
        )

        # Print the response to get the file ID
        print(response)

        # Extract the file ID
        file_id = response['id']
        print(f"The ID of the uploaded file is: {file_id}")
        return file_id


if __name__ == "__main__":
    # Example usage:
    file_loader = FileLoader("dataset/prompt_3.5_train.jsonl", config.OPENAI_API_KEY)
    file_loader.load_files()