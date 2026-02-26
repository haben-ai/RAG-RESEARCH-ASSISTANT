import google.genai as genai

# Create a text generation request
response = genai.generate_text(
    model="gemini-1.0-pro",  # use a valid model from list_models
    prompt="Hello! Summarize this paper for me.",
    temperature=0,
    max_output_tokens=500
)

print(response.text)