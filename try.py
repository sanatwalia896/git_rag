from gitingest import ingest


def process_with_gitingest(github_url):
    # or from URL
    summary, tree, content = ingest(github_url)
    return summary, tree, content


github_url = "https://github.com/sanatwalia896/SQL_LLAMA"

summary, tree, content = process_with_gitingest(github_url=github_url)

print(type(summary))
print(type(tree))

print(type(content))
