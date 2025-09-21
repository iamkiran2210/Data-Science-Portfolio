import ollama

def get_llm_response(prompt: str, model: str = "qwen2:7b") -> str:
    """
    Sends a prompt to the Ollama model and returns the response.

    Args:
        prompt (str): The user's input prompt.
        model (str): The name of the Ollama model to use.

    Returns:
        str: The response from the language model.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )
        return response['message']['content']
    except ollama.ResponseError as e:
        print(f"Ollama API Error: {e.error}")
        return "Sorry, I'm having trouble connecting to the AI model. Please ensure Ollama is running."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred. Please check the logs."
