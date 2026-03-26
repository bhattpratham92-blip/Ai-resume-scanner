import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read job description
with open("job.txt", "r") as f:
    job_desc = f.read()

# Folder where resumes are stored
resume_folder = "resumes"

resumes = []
names = []

# Read all resume files
for file in os.listdir(resume_folder):
    if file.endswith(".txt"):
        with open(os.path.join(resume_folder, file), "r") as f:
            resumes.append(f.read())
            names.append(file)

# Combine job description and resumes
documents = [job_desc] + resumes

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate cosine similarity
similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# Create ranking table
results = pd.DataFrame({
    "Candidate": names,
    "Match Score": similarity_scores
})

# Sort by best match
results = results.sort_values(by="Match Score", ascending=False)

print("\n===== Candidate Ranking =====\n")
print(results)
