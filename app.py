"""Ponto de entrada da interface web: ``streamlit run app.py``."""

import sys
from pathlib import Path

# Permite rodar sem instalar o pacote (pip install -e .).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mlp.ui import main  # noqa: E402

main()
