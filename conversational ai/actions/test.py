"""Small helper to list Google generative AI models.

This module must NOT execute network calls at import time because it
may be imported by the Rasa action server. Run it directly from the
command line to list models instead. The API key is read from
environment variable `GOOGLE_API_KEY`.
"""

import os

def list_generative_models(api_key: str):
    try:
        import google.generativeai as genai
    except Exception:
        print("google.generativeai package not installed")
        return

    if not api_key:
        print("GOOGLE_API_KEY not set in environment; cannot list models")
        return

    genai.configure(api_key=api_key)
    try:
        models = genai.list_models()
        for m in models:
            print(m.name)
    except Exception as e:
        print("Error listing models:", e)


if __name__ == "__main__":
    key = os.getenv("GOOGLE_API_KEY")
    list_generative_models(key)
