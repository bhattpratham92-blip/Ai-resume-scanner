import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SmartHire AI", layout="centered")

st.title("🚀 SmartHire AI – Resume Screening System")

st.write("Enter Job Description and upload resumes to get ranking")

# Job description input
job_desc = st.text_area("📝 Enter Job Description")

# Upload resumes
resume_files = st.file_uploader("📂 Upload Resumes", type=["txt"], accept_multiple_files=True)

# Button
if st.button("🔍 Analyze Resumes"):

    if not job_desc:
        st.warning("Please enter job description ❗")
    
    elif not resume_files:
        st.warning("Please upload resumes ❗")

    else:
        resumes = []
        names = []

        for file in resume_files:
            content = file.read().decode("utf-8")
            resumes.append(content)
            names.append(file.name)

        documents = [job_desc] + resumes

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)

        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        results = pd.DataFrame({
            "Candidate": names,
            "Match Score": similarity_scores
        })

        results = results.sort_values(by="Match Score", ascending=False)

        st.subheader("🏆 Candidate Ranking")
        st.dataframe(results)

        best = results.iloc[0]
        st.success(f"Top Candidate: {best['Candidate']} ({best['Match Score']*100:.2f}%)")
