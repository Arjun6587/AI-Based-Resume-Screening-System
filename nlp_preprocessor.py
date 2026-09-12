import re


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


if __name__ == "__main__":
    sample_text = "Python, Machine Learning & Data Analysis!"

    cleaned = clean_text(sample_text)

    print("Original:")
    print(sample_text)

    print("\nCleaned:")
    print(cleaned)