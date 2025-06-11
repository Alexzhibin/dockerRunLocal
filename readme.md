# Run llm local with docker ##

## Introduction ##
This repository demonstrates how to interact with a locally deployed smollm2 language model using the OpenAI Python client. The example shows a streaming chat conversation where the model shares a happy story with real-time output.

## Requirements ##
1. Python 3.7+
2. OpenAI Python client library
3. Locally running smollm2 service (endpoint: http://localhost:12434/engines/v1)

## Docker Install ##
you need to download and install docker desktop first. please follow this [post](https://www.docker.com/blog/run-llms-locally/)


## Installation ##

```bash
pip install openai
```

## Code Explanation ##
## Client Configuration ##
```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="http://localhost:12434/engines/v1",  # Local model service endpoint
    api_key="docker"  # Authentication key (must match service configuration)
)
```

## Streaming Chat Implementation ##
```python
# Initiate a streaming chat request
completion = client.chat.completions.create(
    model="ai/smollm2",  # Specify model name
    messages=[
        {"role": "system", "content": "Answer the question in a couple sentences."},  # System instruction
        {"role": "user", "content": "Share a happy story with me"}  # User query
    ],
    stream=True  # Enable streaming response
)

# Output response incrementally
for chunk in completion:
    print(chunk.choices[0].delta.content, end="")

```


