import os
try:
    import tomllib
except ImportError:
    import tomli as tomllib

CONFIG_PATH = "config.toml"

with open(CONFIG_PATH, "rb") as f:
    AMP_Config = tomllib.load(f)

AMP_Config["AMP"]["MAIN_LOGS_PATH"] = os.path.expanduser(AMP_Config["AMP"]["MAIN_LOGS_PATH"])
AMP_Config["SERVER"]["PRIVATE_KEY_PATH"] = os.path.expanduser(AMP_Config["SERVER"]["PRIVATE_KEY_PATH"])