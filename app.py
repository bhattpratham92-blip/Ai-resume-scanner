import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="HireNova AI", layout="centered")

# Title
st.title("🚀 HireNova AI – Resume Screening System")
st.write("An AI-powered system to analyze and rank resumes based on job requirements")

# Job Description Input
job_desc = st.text_area("📝 Enter Job Description")

# Upload resumes
resume_files = st.file_uploader(
    "📂 Upload Resumes (.txt)", 
    type=["txt"], 
    accept_multiple_files=True
)

# Button
if st.button("🔍 Analyze Resumes"):

    if not job_desc:
        st.warning("Please enter job description ❗")

    elif not resume_files:
        st.warning("Please upload at least one resume ❗")

    else:
        resumes = []
        names = []

        # Read resumes
        for file in resume_files:
            content = file.read().decode("utf-8")
            resumes.append(content)
            names.append(file.name)

        # Combine JD + resumes
        documents = [job_desc] + resumes

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Cosine Similarity
        similarity_scores = cosine_similarity(
            tfidf_matrix[0:1], 
            tfidf_matrix[1:]
        ).flatten()

        # Create results
        results = pd.DataFrame({
            "Candidate": names,
            "Match Score": similarity_scores
        })

        # Sort results
        results = results.sort_values(by="Match Score", ascending=False)

        # Show table
        st.subheader("🏆 Candidate Ranking")
        st.dataframe(results)

        # Highlight top candidate
        best = results.iloc[0]
        st.success(
            f"Top Candidate: {best['Candidate']} ({best['Match Score']*100:.2f}%)"
        )
