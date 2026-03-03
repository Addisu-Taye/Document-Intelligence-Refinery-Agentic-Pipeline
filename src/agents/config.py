# src/agents/config.py
"""
Configuration Loader for Extraction Rules

Loads thresholds and rules from rubric/extraction_rules.yaml
This enables FDEs to tune behavior without code changes.
"""

import yaml
from pathlib import Path
from typing import Any
from functools import lru_cache


class Config:
    """Singleton configuration loader with caching."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    @lru_cache(maxsize=1)
    def load(cls, config_path: str = "rubric/extraction_rules.yaml") -> dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        cls._config = config
        return config
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-notation key."""
        config = cls.load()
        
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @classmethod
    def get_triage_thresholds(cls) -> dict:
        """Get triage classification thresholds."""
        return cls.load().get('triage', {})
    
    @classmethod
    def get_confidence_thresholds(cls) -> dict:
        """Get confidence scoring thresholds."""
        return cls.load().get('confidence_scoring', {})
    
    @classmethod
    def get_cost_limits(cls) -> dict:
        """Get cost limits for VLM usage."""
        return cls.load().get('cost_limits', {})