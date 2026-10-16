"""Keep test caches and imported app code inside the application boundary."""
import os
from pathlib import Path
import sys

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['MPLCONFIGDIR'] = str(APP / '.cache/matplotlib')
os.environ['TEMP'] = os.environ['TMP'] = str(APP / '.tmp')
