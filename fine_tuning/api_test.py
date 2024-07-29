import os

import openai
from config import OPENAI_API_KEY

# openai.api_key = OPENAI_API_KEY
# # Example of making a request
# response = openai.Completion.create(
#     engine="text-davinci-003",
#     prompt="What are the benefits of eating healthy foods?",
#     max_tokens=50
# )


# Ensure you set your API key
# openai.api_key = os.getenv(OPENAI_API_KEY)
openai.api_key = OPENAI_API_KEY

# Update the method call
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # or another model from the updated API
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the benefits of eating healthy foods?"}
    ]
)

print(response.choices[0].message['content'])