"""Streamlit Application Entrypoint for TEP Optimization Suite.

Runs the master application defined in `streamlit_app.py`.
Allows running either `streamlit run app.py` or `streamlit run streamlit_app.py`.
"""

from pathlib import Path
import runpy

app_path = Path(__file__).parent / "streamlit_app.py"
runpy.run_path(str(app_path), run_name="__main__")
