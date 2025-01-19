from langchain_community.document_loaders import GitLoader
from gitingest import ingest

# # Step 1: Use GitLoader to Load Files from a Repository
# git_loader = GitLoader(
#     repo_path="https://github.com/sanatwalia896/SQL_LLAMA.git",
#     branch="main",  # Optional: Specify the branch
# )

# documents = git_loader.load()

# Step 2: Initialize GitIngest to Manage Repository Content
git_ingest = ingest(repo_url="https://github.com/sanatwalia896/SQL_LLAMA.git")


# Step 4: Query the Ingested Content (Optional)
query_results = git_ingest.query("What does the main function do?")
print(query_results)
