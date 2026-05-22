# MedBot

Flask medical chatbot backed by Pinecone and Hugging Face embeddings.

## Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python store_index.py
python app.py
```

Set `PINECONE_API_KEY` in `.env` before running.

## Deploy on Render

1. Push this folder to a GitHub repository.
2. Go to Render and create a new Blueprint from the repository.
3. Render will read `render.yaml` and create the web service.
4. Add `PINECONE_API_KEY` as an environment variable in Render.
5. Deploy.

The app uses `/healthz` for health checks and runs with Gunicorn in Docker.

## Before Going Public

Rotate any Pinecone API key that has been committed, pasted, or shared. Keep the new key only in `.env` locally and in your deployment provider's environment variables.
