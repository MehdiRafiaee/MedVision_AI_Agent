import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import logging
import json

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, output_dir: str = "output/reports"):
        self.output_dir = output_dir
        
    def generate_model_report(self, model_name: str, metrics: Dict, 
                            history: Dict = None) -> str:
        """Generate comprehensive model evaluation report"""
        try:
            report = {
                'model_name': model_name,
                'timestamp': pd.Timestamp.now().isoformat(),
                'performance_metrics': metrics,
                'training_history': history
            }
            
            # Save JSON report
            report_path = f"{self.output_dir}/{model_name}_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Generate HTML report
            html_report = self._generate_html_report(report)
            html_path = f"{self.output_dir}/{model_name}_report.html"
            with open(html_path, 'w') as f:
                f.write(html_report)
            
            logger.info(f"📄 Report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""
    
    def _generate_html_report(self, report: Dict) -> str:
        """Generate HTML version of the report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report - {report['model_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ margin: 10px 0; padding: 10px; background: #f5f5f5; }}
                .good {{ background: #d4edda; }}
                .warning {{ background: #fff3cd; }}
            </style>
        </head>
        <body>
            <h1>Model Evaluation Report</h1>
            <h2>{report['model_name']}</h2>
            <p>Generated: {report['timestamp']}</p>
            
            <h3>Performance Metrics</h3>
        """
        
        metrics = report['performance_metrics']
        for metric_name, value in metrics.items():
            if metric_name not in ['confusion_matrix', 'classification_report']:
                css_class = 'good' if value > 0.8 else 'warning' if value > 0.6 else ''
                html += f'<div class="metric {css_class}"><strong>{metric_name}:</strong> {value:.4f}</div>'
        
        html += """
            </body>
            </html>
        """
        
        return html
    
    def generate_comparison_report(self, comparison_data: Dict) -> str:
        """Generate comparison report for multiple models"""
        df = pd.DataFrame(comparison_data).T
        
        # Create comparison visualization
        plt.figure(figsize=(12, 8))
        df.plot(kind='bar', figsize=(12, 8))
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_comparison.png')
        
        # Save comparison data
        df.to_csv(f'{self.output_dir}/model_comparison.csv')
        
        return f'{self.output_dir}/model_comparison.png'
