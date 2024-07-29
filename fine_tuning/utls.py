import tiktoken
import numpy as np
import openai


class Utils:
    def __init__(self, dataset, filepath, model="gpt-4o-mini"):
        self.dataset = dataset
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.convo_lens = []
        self.filepath = filepath
        self.model = model

    def num_tokens_from_messages(self, messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def num_assistant_tokens_from_messages(self, messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(self.encoding.encode(message["content"]))
        return num_tokens

    @staticmethod
    def print_distribution(values, name):
        print(f"\n#### Distribution of {name}:")
        print(f"min / max: {min(values)}, {max(values)}")
        print(f"mean / median: {np.mean(values)}, {np.median(values)}")
        print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

    def data_warnings_and_token_counts(self):
        """Data Warnings and Token Counts
        With some lightweight analysis we can identify potential issues in the dataset, like missing messages, and provide statistical insights into message and token counts.

        Missing System/User Messages: Counts the number of conversations missing a "system" or "user" message. Such messages are critical for defining the assistant's behavior and initiating the conversation.
        Number of Messages Per Example: Summarizes the distribution of the number of messages in each conversation, providing insight into dialogue complexity.
        Total Tokens Per Example: Calculates and summarizes the distribution of the total number of tokens in each conversation. Important for understanding fine_tuning costs.
        Tokens in Assistant's Messages: Calculates the number of tokens in the assistant's messages per conversation and summarizes this distribution. Useful for understanding the assistant's verbosity.
        Token Limit Warnings: Checks if any examples exceed the maximum token limit (16,385 tokens), as such examples will be truncated during fine_tuning, potentially resulting in data loss."""
        n_missing_system = 0
        n_missing_user = 0
        n_messages = []
        assistant_message_lens = []

        for ex in self.dataset:
            messages = ex["messages"]
            if not any(message["role"] == "system" for message in messages):
                n_missing_system += 1
            if not any(message["role"] == "user" for message in messages):
                n_missing_user += 1
            n_messages.append(len(messages))
            self.convo_lens.append(self.num_tokens_from_messages(messages))
            assistant_message_lens.append(self.num_assistant_tokens_from_messages(messages))

        print("Num examples missing system message:", n_missing_system)
        print("Num examples missing user message:", n_missing_user)
        self.print_distribution(n_messages, "num_messages_per_example")
        self.print_distribution(self.convo_lens, "num_total_tokens_per_example")
        self.print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
        n_too_long = sum(l > 16385 for l in self.convo_lens)
        print(f"\n{n_too_long} examples may be over the 16,385 token limit, they will be truncated during fine_tuning")

    def cost_estimation(self):
        """Cost Estimation"""
        # Pricing and default n_epochs estimate
        MAX_TOKENS_PER_EXAMPLE = 16385

        TARGET_EPOCHS = 3
        MIN_TARGET_EXAMPLES = 100
        MAX_TARGET_EXAMPLES = 25000
        MIN_DEFAULT_EPOCHS = 1
        MAX_DEFAULT_EPOCHS = 25

        n_epochs = TARGET_EPOCHS
        n_train_examples = len(self.dataset)
        if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
            n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
        elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
            n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

        n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in self.convo_lens)
        print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
        print(f"By default, you'll train for {n_epochs} epochs on this dataset")
        print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")

    def upload(self):
        client = openai
        client.File.create(
            file=open(self.filepath, "rb"),
            purpose="fine-tune")

    def use_trained_model(self):
        client = OpenAI()
        completion = client.chat.completions.create(
            model="ft:gpt-4o-mini:my-org:custom_suffix:id",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
        )
        print(completion.choices[0].message)