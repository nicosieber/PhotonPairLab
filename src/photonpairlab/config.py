from pathlib import Path

# Resolved relative to the installed package (not the repo root), so material data is
# found whether photonpairlab is used from an editable checkout or a built wheel.
RESOURCES_DIR = Path(__file__).parent / "resources"
