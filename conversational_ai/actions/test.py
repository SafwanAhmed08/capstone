import google.generativeai as genai

# Configure your API key
genai.configure(api_key="AIzaSyDWEaQFVVlZ4GwJp--CxXIiUjUlvnkRJUc")

# List all available models
models = genai.list_models()

# Print all model names
for m in models:
    print(m.name)
