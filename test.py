from google import genai

client = genai.Client()

# Get the list of models and their supported methods
models = client.models.list()
for m in models:
    print(m.name)