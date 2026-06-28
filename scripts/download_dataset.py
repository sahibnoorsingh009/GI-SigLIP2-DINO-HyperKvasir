import argparse
from huggingface_hub import snapshot_download
p=argparse.ArgumentParser(); p.add_argument('--repo-id', required=True); p.add_argument('--local-dir', required=True); a=p.parse_args()
snapshot_download(repo_id=a.repo_id, repo_type='dataset', local_dir=a.local_dir, resume_download=True)
print('Downloaded to', a.local_dir)
