import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# File path to the Excel file
file_path = "../quote-extraction/causal-reasoning-quotes.xlsx"  # Replace with your file path

# Read the Excel file
data = pd.read_excel(file_path, engine='openpyxl')

# Extract the "Circumstance" column
circumstances = data["Circumstance"].dropna()  # Drop rows with missing values

# Convert the column to a list of sentences
sentences = circumstances.tolist()

# Step 1: Generate sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other models as well
embeddings = model.encode(sentences)

# Step 2: Cluster the embeddings
num_clusters = 10  # Choose the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(embeddings)
labels = kmeans.labels_

# Step 3: Group sentences by clusters
clusters = {}
for sentence, label in zip(sentences, labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(sentence)

# Step 4: Write clusters to a Markdown file
markdown_file = "clusters.md"
with open(markdown_file, "w") as file:
    file.write("# Sentence Clusters\n\n")
    for cluster_id, cluster_sentences in clusters.items():
        file.write(f"## Cluster {cluster_id}\n")
        for sentence in cluster_sentences:
            file.write(f"- {sentence}\n")
        file.write("\n")

print(f"Clusters have been written to {markdown_file}.")
