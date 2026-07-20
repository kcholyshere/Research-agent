FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# ADK's dev UI by default; the optional Streamlit UI (src/ui/app.py) is an
# alternative entrypoint - swap the CMD below or run both containers. Verify
# in-container data access before relying on this (Finrag audit A3: the image
# ships no data/ or models/ - mount them).
EXPOSE 8000
CMD ["adk", "web", "src", "--host", "0.0.0.0"]
