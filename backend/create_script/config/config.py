import os, json
from pathlib import Path

class Settings :
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_SECRET_DIR = os.path.join(BASE_DIR, ".config_secret")
    CONFIG_SECRET_COMMON_FILE = os.path.join(CONFIG_SECRET_DIR, "setting_local.json")

    config_secret_common = json.loads(open(CONFIG_SECRET_COMMON_FILE).read())

    OPENAI_API_KEY: str = config_secret_common["openai_api_key"]
    
settings = Settings()