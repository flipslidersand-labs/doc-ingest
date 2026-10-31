"""Load .env as an import side effect.

Several modules read environment variables at import time, so this module must be
imported before them (see the top of cli.py). Keeping it in its own module lets
cli.py keep all imports at the top of the file without E402.
"""

from dotenv import load_dotenv

load_dotenv()
