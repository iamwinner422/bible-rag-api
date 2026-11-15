import importlib
import sys
from pathlib import Path


def load_search_engine(project: Path):
    project = project.resolve()
    sys.path.insert(0, str(project))
    if (project / "app" / "search" / "engine.py").is_file():
        module = importlib.import_module("app.search.engine")
    else:
        module = importlib.import_module("search_engine")
    return module.SearchEngine
