from collections import defaultdict
from check_dataset_format import CheckDatasetFormat

"""====Reference: https://cookbook.openai.com/examples/chat_finetuning_data_prep ============================"""


class CheckDatasetFormatFor3P5(CheckDatasetFormat):
    def __init__(self, filepath="../dataset/prompt_3.5_train.jsonl"):
        super().__init__(file_path=filepath)

    def error_check(self):
        """
        Error checks for the 3.5 dataset format.
        Prints error messages if any errors are found.
        :return:
        """
        format_errors = defaultdict(int)

        for ex in self.dataset:
            if not isinstance(ex, dict):
                format_errors["data_type"] += 1
                continue

            messages = ex.get("messages", None)
            if not messages:
                format_errors["missing_messages_list"] += 1
                continue

            for message in messages:
                if "role" not in message or "content" not in message:
                    format_errors["message_missing_key"] += 1

                if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                    format_errors["message_unrecognized_key"] += 1

                if message.get("role", None) not in ("system", "user", "assistant", "function"):
                    format_errors["unrecognized_role"] += 1

                content = message.get("content", None)
                function_call = message.get("function_call", None)

                if (not content and not function_call) or not isinstance(content, str):
                    format_errors["missing_content"] += 1

            if not any(message.get("role", None) == "assistant" for message in messages):
                format_errors["example_missing_assistant_message"] += 1

        if format_errors:
            print("Found errors:")
            for k, v in format_errors.items():
                print(f"{k}: {v}")
        else:
            print("No errors found")


if __name__ == "__main__":
    check_dataset = CheckDatasetFormatFor3P5()
    check_dataset.error_check()
