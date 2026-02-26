import google.genai as genai

models = genai.list_models()
for m in models:
    print(m.name, m.supported_methods)