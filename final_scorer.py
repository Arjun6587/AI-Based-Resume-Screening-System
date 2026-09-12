from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from resume_parser import extract_resume_text
from nlp_preprocessor import clean_text
from skill_matcher import analyze_skills


# Load Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------------
# Job Description
# -----------------------------------------

job_description = extract_resume_text(
    "sample_data/job_description.txt"
)

cleaned_job = clean_text(job_description)

job_embedding = model.encode([cleaned_job])


# -----------------------------------------
# Candidates
# -----------------------------------------

candidates = [
    ("ARUN KUMAR", "sample_data/candidate_1.txt"),
    ("PRIYA MENON", "sample_data/candidate_2.txt"),
    ("RAHUL NAIR", "sample_data/candidate_3.txt")
]


results = []


# -----------------------------------------
# Analyze each candidate
# -----------------------------------------

for name, file_path in candidates:

    resume = extract_resume_text(file_path)

    cleaned_resume = clean_text(resume)


    # -------------------------------------
    # 1. Semantic Similarity
    # -------------------------------------

    resume_embedding = model.encode([cleaned_resume])

    similarity = cosine_similarity(
        resume_embedding,
        job_embedding
    )[0][0]

    semantic_score = similarity * 100


    # -------------------------------------
    # 2. Skill Analysis
    # -------------------------------------

    skill_results = analyze_skills(resume)

    matched_required = skill_results["matched_required"]
    missing_required = skill_results["missing_required"]

    matched_preferred = skill_results["matched_preferred"]
    missing_preferred = skill_results["missing_preferred"]


    # Required skill percentage
    required_score = (
        len(matched_required) / 7
    ) * 100


    # Preferred skill percentage
    preferred_score = (
        len(matched_preferred) / 3
    ) * 100


    # Combined skill score
    skill_score = (
        required_score * 0.70
        + preferred_score * 0.30
    )


    # -------------------------------------
    # 3. Experience
    # -------------------------------------

    if "experience" in cleaned_resume or "intern" in cleaned_resume:
        experience_score = 100
    else:
        experience_score = 0


    # -------------------------------------
    # 4. Education
    # -------------------------------------

    if (
        "bachelor" in cleaned_resume
        or "computer science" in cleaned_resume
        or "computer applications" in cleaned_resume
    ):
        education_score = 100
    else:
        education_score = 0


    # -------------------------------------
    # 5. Final Weighted Score
    # -------------------------------------

    final_score = (
        semantic_score * 0.60
        + skill_score * 0.25
        + experience_score * 0.10
        + education_score * 0.05
    )


    # -------------------------------------
    # Store Results
    # -------------------------------------

    results.append({
        "name": name,
        "semantic": semantic_score,
        "required": required_score,
        "preferred": preferred_score,
        "skills": skill_score,
        "experience": experience_score,
        "education": education_score,
        "final": final_score,
        "matched_required": matched_required,
        "missing_required": missing_required,
        "matched_preferred": matched_preferred,
        "missing_preferred": missing_preferred
    })


# -----------------------------------------
# Rank Candidates
# -----------------------------------------

results.sort(
    key=lambda x: x["final"],
    reverse=True
)


# -----------------------------------------
# Display Results
# -----------------------------------------

print("\n==========================================")
print("     AI RESUME SCREENING FINAL RESULTS")
print("==========================================\n")


for rank, result in enumerate(results, start=1):

    print(f"{rank}. {result['name']}")

    print(
        f"   Semantic Score   : "
        f"{result['semantic']:.2f}%"
    )

    print(
        f"   Required Skills  : "
        f"{result['required']:.2f}%"
    )

    print(
        f"   Preferred Skills : "
        f"{result['preferred']:.2f}%"
    )

    print(
        f"   Overall Skill    : "
        f"{result['skills']:.2f}%"
    )

    print(
        f"   Experience       : "
        f"{result['experience']:.2f}%"
    )

    print(
        f"   Education        : "
        f"{result['education']:.2f}%"
    )

    print(
        f"   FINAL SCORE      : "
        f"{result['final']:.2f}%"
    )


    print("\n   Matched Required Skills:")

    for skill in result["matched_required"]:
        print(f"      ✓ {skill}")


    if result["missing_required"]:

        print("\n   Missing Required Skills:")

        for skill in result["missing_required"]:
            print(f"      ✗ {skill}")


    print("\n   Matched Preferred Skills:")

    for skill in result["matched_preferred"]:
        print(f"      ✓ {skill}")


    if result["missing_preferred"]:

        print("\n   Missing Preferred Skills:")

        for skill in result["missing_preferred"]:
            print(f"      ✗ {skill}")


    print("\n" + "-" * 42 + "\n")