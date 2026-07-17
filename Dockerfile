FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# ADK's dev UI; swap for the Streamlit/Gradio entrypoint if the optional UI
# lands. Verify in-container data access before relying on this (Finrag audit
# A3: the image ships no data/ or models/ - mount them).
EXPOSE 8000
CMD ["adk", "web", "src", "--host", "0.0.0.0"]
