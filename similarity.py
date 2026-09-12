from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from resume_parser import extract_resume_text
from nlp_preprocessor import clean_text


# Load Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Read the job description
job_description = extract_resume_text(
    "sample_data/job_description.txt"
)

# Read Candidate 1 resume
candidate_resume = extract_resume_text(
    "sample_data/candidate_1.txt"
)


# Clean both texts
job_description = clean_text(job_description)
candidate_resume = clean_text(candidate_resume)


# Generate embeddings
job_embedding = model.encode([job_description])
candidate_embedding = model.encode([candidate_resume])


# Calculate cosine similarity
similarity = cosine_similarity(
    candidate_embedding,
    job_embedding
)[0][0]


# Convert to percentage
score = similarity * 100


print("Candidate 1: ARUN KUMAR")
print("Job: Machine Learning Engineer")
print(f"Similarity Score: {score:.2f}%")