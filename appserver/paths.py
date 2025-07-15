# aidoc-task-1/paths.py
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 资源目录
RESOURCES_DIR = PROJECT_ROOT / "resources"

SRC_DIR = PROJECT_ROOT / "appserver"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"