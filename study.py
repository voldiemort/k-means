"""Run the local checkout without first installing the package."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
# Optional local dependency directory; ordinary installations do not need it.
if (ROOT / ".local-deps").is_dir():
    sys.path.insert(0, str(ROOT / ".local-deps"))

from kmeans_study.cli import main

if __name__ == "__main__":
    main()
