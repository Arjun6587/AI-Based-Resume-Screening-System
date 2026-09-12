# TalentScreen AI — AI Resume Screening System

TalentScreen AI is an AI-based resume screening and candidate ranking system that helps recruiters compare candidate resumes against a job description. The system uses Natural Language Processing (NLP), Transformer-based text embeddings, skill matching, experience analysis, and education analysis to generate candidate rankings and explainable screening results.

## Project Overview

Recruiters often need to manually review a large number of resumes for a single job opening. This process can be time-consuming and difficult to perform consistently.

TalentScreen AI automates the initial resume screening process by analyzing resumes against a given Job Description (JD). It identifies relevant skills, evaluates experience and education relevance, calculates an overall compatibility score, and ranks candidates according to their suitability for the job.

The system is designed as a screening-support tool and does not replace human recruitment decisions.

## Main Objectives

- Automate the initial resume screening process
- Compare resumes with job descriptions
- Identify relevant candidate skills
- Measure semantic similarity between resumes and job descriptions
- Evaluate experience relevance
- Evaluate education relevance
- Rank candidates according to their overall compatibility
- Provide explainable insights into candidate strengths and gaps
- Handle multiple resumes efficiently
- Detect duplicate, empty, and corrupted resume files
- Export screening results for further analysis

## Key Features

### Job Description Input

The system provides two ways to provide a Job Description:

- Upload a PDF, DOCX, or TXT file
- Enter the Job Description manually

### Multiple Resume Upload

Recruiters can upload multiple candidate resumes at the same time.

Supported formats:

- PDF
- DOCX
- TXT

### Resume Preview

Before screening, the system extracts and displays the text from uploaded resumes so that users can verify the extracted content.

### Duplicate Resume Detection

The system detects duplicate resumes based on their extracted resume content.

If the same resume is uploaded with different filenames, it can still be recognized as a duplicate.

### Empty and Invalid File Detection

The system checks uploaded files before AI analysis.

It can identify:

- Empty resumes
- Resumes with no readable text
- Corrupted PDF or DOCX files
- Invalid files
- Unsupported file formats

Invalid or unusable files are skipped without stopping the screening process.

## AI and NLP Methodology

The system uses a pretrained Sentence Transformer model:

`all-MiniLM-L6-v2`

The model converts text into numerical vector representations called embeddings.

Each embedding contains 384 numerical dimensions.

The Job Description and each resume are converted into embeddings and compared using Cosine Similarity.

### Processing Pipeline

```text
Job Description
       ↓
Text Extraction
       ↓
Text Preprocessing
       ↓
Sentence Transformer
       ↓
384-Dimensional Embedding
       ↓
Cosine Similarity
       ↓
Semantic Match Score
       ↓
Skill Analysis
       ↓
Experience Analysis
       ↓
Education Analysis
       ↓
Weighted Final Score
       ↓
Candidate Ranking
       ↓
Explainable Results
```