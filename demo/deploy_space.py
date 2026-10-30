"""
Upload build/web/ to a FREE static Hugging Face Space (creates it the first time).
The simulator runs in the visitor's browser, so no paid hardware is needed.

    python demo/deploy_space.py                      # -> <your-username>/novaflow-demo
    python demo/deploy_space.py USERNAME/SPACE-NAME  # custom name

Needs: pip install huggingface_hub, and a Hugging Face access token with WRITE
permission (huggingface.co -> Settings -> Access Tokens).
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi, login

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "build" / "web"


def main() -> int:
    if not (SPACE / "index.html").exists():
        print("build/web/ not found. Run first:  python demo/build_web.py")
        return 1
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        print("Not logged in. Paste a Hugging Face token with WRITE permission:")
        login()
        user = api.whoami()["name"]
    repo_id = sys.argv[1] if len(sys.argv) > 1 else f"{user}/novaflow-demo"
    print(f"Deploying to Space {repo_id} ...")
    api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=str(SPACE), repo_id=repo_id, repo_type="space",
                      commit_message="Deploy NovaFlow live demo")
    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nDone. Your live demo (ready within a minute) is at:\n  {url}")
    print("Add this link to README.md (Project updates) for Step 10.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
