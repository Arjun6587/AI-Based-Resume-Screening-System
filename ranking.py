from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from resume_parser import extract_resume_text
from nlp_preprocessor import clean_text


# Load Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Read and clean the job description
job_description = extract_resume_text(
    "sample_data/job_description.txt"
)

job_description = clean_text(job_description)

# Generate job description embedding
job_embedding = model.encode([job_description])


# Candidate files
candidates = [
    ("ARUN KUMAR", "sample_data/candidate_1.txt"),
    ("PRIYA MENON", "sample_data/candidate_2.txt"),
    ("RAHUL NAIR", "sample_data/candidate_3.txt")
]


results = []


# Compare every candidate with the job
for name, file_path in candidates:

    resume_text = extract_resume_text(file_path)
    resume_text = clean_text(resume_text)

    resume_embedding = model.encode([resume_text])

    similarity = cosine_similarity(
        resume_embedding,
        job_embedding
    )[0][0]

    score = similarity * 100

    results.append((name, score))


# Sort candidates from highest score to lowest
results.sort(key=lambda x: x[1], reverse=True)


# Display ranking
print("\n===================================")
print(" AI RESUME SCREENING RESULTS")
print("===================================\n")

for rank, (name, score) in enumerate(results, start=1):

    print(f"{rank}. {name}")
    print(f"   Match Score: {score:.2f}%")
    print()