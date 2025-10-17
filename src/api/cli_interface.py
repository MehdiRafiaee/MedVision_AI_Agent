import argparse
import sys
from pathlib import Path
import logging

from ..models.ensemble_model import EnsembleMedicalModel
from ..utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class CLIInterface:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        self.ensemble_model = None
    
    def train_command(self, args):
        """Handle train command"""
        logger.info(f"Starting training with data from: {args.data_dir}")
        
        # Initialize ensemble model
        self.ensemble_model = EnsembleMedicalModel(self.config)
        self.ensemble_model.initialize_models()
        
        # TODO: Implement training logic
        print("Training started...")
        
    def predict_command(self, args):
        """Handle predict command"""
        if not Path(args.image_path).exists():
            logger.error(f"Image not found: {args.image_path}")
            return
        
        # Load model if not already loaded
        if self.ensemble_model is None:
            self.ensemble_model = EnsembleMedicalModel(self.config)
            # TODO: Load trained model
        
        # TODO: Implement prediction logic
        print(f"Predicting on: {args.image_path}")
        
    def serve_command(self, args):
        """Handle serve command"""
        from .rest_api import start_rest_api
        start_rest_api(host=args.host, port=args.port, config=self.config)
    
    def run(self):
        """Main CLI entry point"""
        parser = argparse.ArgumentParser(description="MedVision AI Agent CLI")
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Train command
        train_parser = subparsers.add_parser("train", help="Train models")
        train_parser.add_argument("--data_dir", required=True, help="Training data directory")
        train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
        
        # Predict command  
        predict_parser = subparsers.add_parser("predict", help="Predict on image")
        predict_parser.add_argument("--image_path", required=True, help="Path to image")
        predict_parser.add_argument("--model_type", default="ensemble", 
                                  help="Model type to use")
        
        # Serve command
        serve_parser = subparsers.add_parser("serve", help="Start API server")
        serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
        serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
        
        args = parser.parse_args()
        
        if args.command == "train":
            self.train_command(args)
        elif args.command == "predict":
            self.predict_command(args)
        elif args.command == "serve":
            self.serve_command(args)
        else:
            parser.print_help()

if __name__ == "__main__":
    cli = CLIInterface()
    cli.run()
