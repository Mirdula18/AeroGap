"""Deploy the API to a Hugging Face Docker Space (free CPU tier, no billing).

Stages api/ + the grid Parquet files into a temp folder and uploads it.
Needs a logged-in Hugging Face token (`hf auth login`); data never goes to git.

    python -m api.deploy_hf                      # -> <user>/aerogap-api
    python -m api.deploy_hf --space aerogap-api --res 7
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: AeroGap API
emoji: 🛰️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
license: mit
short_description: H3 grid of India with distance to nearest AQ monitor
---

# AeroGap API

Predicting the pollution your monitors miss. Source: https://github.com/joedanields/AeroGap

- `GET /grid?bbox=minLon,minLat,maxLon,maxLat[&res=4..{res}]`
- `GET /stations[?bbox=...]`
- `GET /health` and OpenAPI docs at `/docs`

Station data: OpenAQ (CC BY 4.0). Boundaries: geoBoundaries (CC BY 2.5 IN / ODbL).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", default="aerogap-api")
    parser.add_argument("--res", type=int, default=7)
    args = parser.parse_args()

    api = HfApi()
    repo_id = f"{api.whoami()['name']}/{args.space}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copytree(ROOT / "api", stage / "api", ignore=shutil.ignore_patterns("__pycache__", "deploy_hf.py"))
        shutil.move(stage / "api" / "Dockerfile", stage / "Dockerfile")
        (stage / "data" / "grid").mkdir(parents=True)
        for name in (f"grid_r{args.res}_api.parquet", "stations.parquet"):
            shutil.copy2(ROOT / "data" / "grid" / name, stage / "data" / "grid" / name)
        (stage / "README.md").write_text(SPACE_README.format(res=args.res), encoding="utf-8")
        api.upload_folder(folder_path=stage, repo_id=repo_id, repo_type="space",
                          commit_message="Deploy AeroGap API")

    user, name = repo_id.split("/")
    print(f"Space:    https://huggingface.co/spaces/{repo_id}")
    print(f"API base: https://{user.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    main()
