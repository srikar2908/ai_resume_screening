import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""  # Avoid NoneType error
        return text.strip()  # Remove extra spaces
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Combine job description with resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Streamlit app
st.title("📄 AIRecruit Pro: 🤖 Intelligent Resume Screening & 🔝 Candidate Ranking")

# Job description input
st.header("📌 Job Description")
job_description = st.text_area("Enter the job description", height=150)

# File uploader
st.header("📤 Upload Resumes (PDF)")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("📊 Ranking Resumes")
    
    resumes = []
    valid_files = []
    
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        if text.startswith("Error reading PDF"):  # Skip corrupted files
            st.warning(f"⚠️ Skipping {file.name}: {text}")
        else:
            resumes.append(text)
            valid_files.append(file.name)

    # Ensure there is at least one valid resume
    if resumes:
        # Rank resumes
        scores = rank_resumes(job_description, resumes)

        # Display scores
        results = pd.DataFrame({"Resume": valid_files, "Score": scores})
        results = results.sort_values(by="Score", ascending=False)

        st.dataframe(results)  # More interactive than st.write()
    else:
        st.error("❌ No valid resumes found. Please upload valid PDF files.")
