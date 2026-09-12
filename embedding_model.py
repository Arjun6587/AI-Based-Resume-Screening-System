from sentence_transformers import SentenceTransformer


# Load the pre-trained Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embedding(text):
    """Convert text into a numerical embedding."""
    embedding = model.encode(text)

    return embedding


if __name__ == "__main__":
    sample_text = "Python machine learning data analysis"

    embedding = generate_embedding(sample_text)

    print("Embedding generated successfully!")
    print("Embedding size:", len(embedding))
    print("First 10 values:")
    print(embedding[:10])