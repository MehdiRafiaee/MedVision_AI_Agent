import yaml
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"✅ Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any):
        """Update configuration value"""
        keys = key.split('.')
        config_ref = self.config
        
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        config_ref[keys[-1]] = value
        logger.info(f"🔧 Configuration updated: {key} = {value}")
    
    def save_config(self, filepath: str = None):
        """Save current configuration to file"""
        save_path = filepath or self.config_path
        
        try:
            with open(save_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            logger.info(f"💾 Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"❌ Error saving configuration: {e}")
    
    def validate_config(self) -> bool:
        """Validate configuration structure"""
        required_sections = ['model', 'training', 'features', 'evaluation']
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"❌ Missing required configuration section: {section}")
                return False
        
        logger.info("✅ Configuration validation passed")
        return True
