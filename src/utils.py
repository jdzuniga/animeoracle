from pathlib import Path
from datetime import datetime


def get_latest_dated_folder(base_path: str | Path) -> str:
    base = Path(base_path)
    dated_folders = []

    for p in base.iterdir():
        if p.is_dir():
            date = datetime.strptime(p.name, "%Y-%m-%d")
            dated_folders.append(date)

    latest_folder = max(dated_folders)
    latest_date = str(latest_folder.date())
    return latest_date
