from openai import OpenAI
import os

client = OpenAI(
    base_url = "http://localhost:12434/engines/v1",
    api_key = "docker"
)

##not stream##
completion = client.chat.completions.create(
    model="ai/smollm2",
    messages=[
        {"role": "system", "content": "Answer the question in a couple sentences."},
        {"role": "user", "content": "Share a happy story with me"}
    ]
)

print(completion.choices[0].message.content)














