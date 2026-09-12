from resume_parser import extract_resume_text
from nlp_preprocessor import clean_text


# ============================================================
# COMMON SKILL DATABASE
# ============================================================

SKILL_DATABASE = [
    # Programming
    "python",
    "java",
    "c++",
    "c",
    "javascript",
    "typescript",
    "php",

    # Machine Learning / AI
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "natural language processing",
    "computer vision",
    "tensorflow",
    "pytorch",
    "scikit learn",

    # Data Science
    "data analysis",
    "data science",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "data visualization",
    "statistics",

    # Databases
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "oracle",

    # Web Development
    "html",
    "css",
    "react",
    "node js",
    "django",
    "flask",
    "spring boot",

    # Cloud / Tools
    "aws",
    "azure",
    "google cloud",
    "docker",
    "kubernetes",
    "git",
    "github",

    # Other
    "power bi",
    "excel",
    "tableau"
]


# ============================================================
# EXTRACT SKILLS FROM JOB DESCRIPTION
# ============================================================

def extract_job_skills(job_description):

    cleaned_job = clean_text(job_description)

    detected_skills = []

    for skill in SKILL_DATABASE:

        if skill in cleaned_job:

            detected_skills.append(skill)

    return detected_skills


# ============================================================
# FIND MATCHING SKILLS
# ============================================================

def find_matching_skills(resume_text, skill_list):

    cleaned_resume = clean_text(resume_text)

    matched_skills = []

    for skill in skill_list:

        if skill in cleaned_resume:

            matched_skills.append(skill)

    return matched_skills


# ============================================================
# ANALYZE RESUME AGAINST JOB DESCRIPTION
# ============================================================

def analyze_skills(resume_text, job_description):

    # --------------------------------------------------------
    # Extract skills from job description
    # --------------------------------------------------------

    job_skills = extract_job_skills(
        job_description
    )


    # --------------------------------------------------------
    # If no skills detected
    # --------------------------------------------------------

    if not job_skills:

        return {
            "job_skills": [],
            "matched_skills": [],
            "missing_skills": []
        }


    # --------------------------------------------------------
    # Find matching skills in resume
    # --------------------------------------------------------

    matched_skills = find_matching_skills(
        resume_text,
        job_skills
    )


    # --------------------------------------------------------
    # Find missing skills
    # --------------------------------------------------------

    missing_skills = [

        skill
        for skill in job_skills
        if skill not in matched_skills

    ]


    return {

        "job_skills": job_skills,

        "matched_skills": matched_skills,

        "missing_skills": missing_skills

    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    job_description = """
    Machine Learning Engineer

    Required Skills:
    Python, Machine Learning, Data Analysis,
    Pandas, NumPy, Scikit-learn and SQL.

    Preferred Skills:
    Deep Learning, TensorFlow and Data Visualization.
    """


    resume = extract_resume_text(
        "sample_data/candidate_1.txt"
    )


    results = analyze_skills(
        resume,
        job_description
    )


    print("\nDetected Job Skills:")

    for skill in results["job_skills"]:

        print("•", skill)


    print("\nMatched Skills:")

    for skill in results["matched_skills"]:

        print("✓", skill)


    print("\nMissing Skills:")

    for skill in results["missing_skills"]:

        print("✗", skill)