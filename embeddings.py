from google import genai
from config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)


def generate_embeddings(chunks: list[str]) -> list[list[float]]:
    embeddings = []

    for text in chunks:
        result = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text
        )

        embeddings.append(result.embeddings[0].values)

    return embeddings