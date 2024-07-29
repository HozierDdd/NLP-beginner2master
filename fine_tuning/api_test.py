import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY
# Example of making a request
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="What are the benefits of eating healthy foods?",
    max_tokens=50
)