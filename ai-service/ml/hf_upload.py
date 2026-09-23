from __future__ import annotations

import argparse
from huggingface_hub import HfApi

def main() -> None:
    parser = argparse.ArgumentParser(description='Upload a ClinevoOne-Frontier export to the Hugging Face Hub')
    parser.add_argument('--repo-id', default='Aravindhan11/cli-ai')
    parser.add_argument('--folder', default='/cache/clinevo-owned/hub')
    parser.add_argument('--private', action='store_true')
    args = parser.parse_args()
    api = HfApi()
    api.create_repo(args.repo_id, repo_type='model', private=args.private, exist_ok=True)
    url = api.upload_folder(repo_id=args.repo_id, folder_path=args.folder, repo_type='model', commit_message='Publish ClinevoOne-Frontier research artifact')
    print(url)

if __name__ == '__main__':
    main()
