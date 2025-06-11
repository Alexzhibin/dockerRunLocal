from openai import OpenAI
import os
import json
import time
import pandas as pd

# Initialize the client
client = OpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key="docker"
)


# Example 1: Basic call - Non-streaming response
def basic_chat():
    completion = client.chat.completions.create(
        model="ai/smollm2",
        messages=[
            {"role": "system", "content": "You are a friendly and helpful assistant."},
            {"role": "user", "content": "Share an interesting fact with me."}
        ]
    )
    print("Basic call result:")
    print(completion.choices[0].message.content)
    return completion


# Example 2: Streaming response - Display generated content in real-time
def stream_chat():
    print("\nStreaming response result:")
    response = client.chat.completions.create(
        model="ai/smollm2",
        messages=[
            {"role": "system", "content": "You are a detailed explainer."},
            {"role": "user", "content": "Explain the basic principles of quantum computing."}
        ],
        stream=True
    )

    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            print(content, end="", flush=True)
    print("\n")
    return full_response


# Example 3: Function calling - Structured output
def function_call():
    # Define function descriptions
    functions = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a specified city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["city"]
            }
        }
    ]

    # Call the API
    response = client.chat.completions.create(
        model="ai/smollm2",
        messages=[
            {"role": "user", "content": "What's the weather like in Beijing today?"}
        ],
        functions=functions
    )

    # Parse the function call
    function_call = response.choices[0].message.function_call
    if function_call:
        function_name = function_call.name
        function_args = json.loads(function_call.arguments)
        print(f"Function call: {function_name}")
        print(f"Arguments: {function_args}")

        # Here you can execute the actual function based on the name and arguments
        # For demonstration purposes, we'll mock a response
        mock_response = {
            "city": function_args["city"],
            "temperature": 25,
            "unit": "celsius",
            "condition": "Sunny"
        }

        # Send the function result back to the model
        second_response = client.chat.completions.create(
            model="ai/smollm2",
            messages=[
                {"role": "user", "content": "What's the weather like in Beijing today?"},
                response.choices[0].message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(mock_response)
                }
            ]
        )

        print("Final answer:")
        print(second_response.choices[0].message.content)
        return second_response
    else:
        print("No function call")
        return response


# Example 4: Handling context - Multi-turn conversation
def conversation_with_context():
    # Initialize conversation history
    messages = [
        {"role": "system", "content": "You are a history knowledge expert."}
    ]

    # Add the user's first question
    messages.append({"role": "user", "content": "Who invented the light bulb?"})

    # Send the first request
    response = client.chat.completions.create(
        model="ai/smollm2",
        messages=messages
    )

    # Save the assistant's reply
    assistant_message = response.choices[0].message
    messages.append(assistant_message)

    print("Question 1: Who invented the light bulb?")
    print("Answer 1:", assistant_message.content)

    # Add the user's second question
    messages.append({"role": "user", "content": "What other important inventions did he make?"})

    # Send the second request
    response = client.chat.completions.create(
        model="ai/smollm2",
        messages=messages
    )

    # Save the assistant's reply
    assistant_message = response.choices[0].message
    messages.append(assistant_message)

    print("\nQuestion 2: What other important inventions did he make?")
    print("Answer 2:", assistant_message.content)

    return messages


# Example 5: Batch processing - Read data from CSV and generate responses
def batch_processing():
    # Create example data
    data = {
        "Question": [
            "Explain the basic concepts of machine learning",
            "Recommend three science fiction novels worth reading",
            "How to improve Python programming skills",
            "What is blockchain technology"
        ]
    }

    df = pd.DataFrame(data)

    # Process each question
    responses = []
    for index, row in df.iterrows():
        print(f"\nProcessing question {index + 1}: {row['Question']}")

        response = client.chat.completions.create(
            model="ai/smollm2",
            messages=[
                {"role": "system", "content": "You are a concise information provider."},
                {"role": "user", "content": row["Question"]}
            ]
        )

        answer = response.choices[0].message.content
        responses.append(answer)
        print(f"Answer: {answer[:100]}...")  # Print only the first 100 characters

        # Add a delay to avoid making requests too quickly
        time.sleep(1)

    # Add the answers to the DataFrame
    df["Answer"] = responses

    # Save the results
    df.to_csv("llm_responses.csv", index=False, encoding="utf-8")
    print("\nResults saved to llm_responses.csv")
    return df


# Example 6: Custom parameters - Control output quality and diversity
def custom_parameters():
    # High quality, low diversity
    print("\nHigh quality, low diversity:")
    response = client.chat.completions.create(
        model="ai/smollm2",
        messages=[
            {"role": "user", "content": "Write a short poem about spring"}
        ],
        temperature=0.2,  # Low temperature = low diversity
        top_p=0.9,
        max_tokens=100,
        frequency_penalty=0.5,
        presence_penalty=0.0
    )
    print(response.choices[0].message.content)

    # High creativity, high diversity
    print("\nHigh creativity, high diversity:")
    response = client.chat.completions.create(
        model="ai/smollm2",
        messages=[
            {"role": "user", "content": "Write a short poem about spring"}
        ],
        temperature=0.8,  # High temperature = high diversity
        top_p=0.95,
        max_tokens=100,
        frequency_penalty=0.0,
        presence_penalty=0.3
    )
    print(response.choices[0].message.content)

    return response


# Example 7: Error handling
def error_handling():
    try:
        # Intentionally use a non-existent model name
        response = client.chat.completions.create(
            model="nonexistent-model",
            messages=[
                {"role": "user", "content": "This will fail"}
            ]
        )
    except Exception as e:
        print(f"\nError handling example:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")

        # Handle different types of errors
        if "model" in str(e).lower():
            print("Error: Model does not exist. Please check the model name.")
        elif "API" in str(e).lower():
            print("Error: API connection issue. Please check your API key and URL.")
        else:
            print("Error: Unknown issue")

        # You can return a default response or implement retry logic
        return "Sorry, I encountered an error and cannot process your request."

    return response


# Execute all examples
if __name__ == "__main__":
    print("=== Example 1: Basic Call ===")
    basic_chat()

    print("\n=== Example 2: Streaming Response ===")
    stream_chat()

    print("\n=== Example 3: Function Calling ===")
    function_call()

    print("\n=== Example 4: Multi-turn Conversation ===")
    conversation_with_context()

    print("\n=== Example 5: Batch Processing ===")
    batch_processing()

    print("\n=== Example 6: Custom Parameters ===")
    custom_parameters()

    print("\n=== Example 7: Error Handling ===")
    error_handling()
