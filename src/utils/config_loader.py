"""YAML configuration loader with environment variable override support."""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class ConfigLoader:
    """
    Loads and provides access to the central YAML config.
    Supports environment variable overrides for any key.
    """

    _instance: Optional["ConfigLoader"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: Optional[str] = None) -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if self._loaded:
            return
        if config_path is None:
            # Resolve relative to project root
            root = Path(__file__).resolve().parents[2]
            config_path = root / "config" / "config.yaml"
        self._load(config_path)
        self._loaded = True

    def _load(self, path: Path) -> None:
        """Load YAML configuration from disk."""
        with open(path, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value using dot-notation key.
        e.g., get("data.raw_data_path")
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        # Allow ENV override: DATA__RAW_DATA_PATH
        env_key = key.replace(".", "__").upper()
        return os.getenv(env_key, value)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Return a full section of the config as a dict."""
        return self._config.get(section, {})

    @property
    def config(self) -> Dict[str, Any]:
        return self._config


# Singleton instance
cfg = ConfigLoader()
