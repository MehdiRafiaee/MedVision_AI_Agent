# src/models/traditional_ml.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

class TraditionalMLModels:
    def __init__(self):
        self.models = {
            'random_forest': Pipeline([
                ('scaler', StandardScaler()),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            'svm': Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', probability=True, random_state=42))
            ]),
            'logistic': Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(random_state=42))
            ])
        }
        
        self.trained_models = {}
    
    def train(self, X, y, model_name='all'):
        """آموزش مدل‌های کلاسیک"""
        if model_name == 'all':
            for name, model in self.models.items():
                print(f"Training {name}...")
                model.fit(X, y)
                self.trained_models[name] = model
        else:
            if model_name in self.models:
                self.models[model_name].fit(X, y)
                self.trained_models[model_name] = self.models[model_name]
            else:
                raise ValueError(f"Model {model_name} not found")
    
    def predict(self, X, model_name):
        """پیش‌بینی با مدل آموزش دیده"""
        if model_name in self.trained_models:
            return self.trained_models[model_name].predict_proba(X)
        else:
            raise ValueError(f"Model {model_name} not trained")
    
    def get_feature_importance(self, model_name='random_forest'):
        """دریافت اهمیت ویژگی‌ها (برای Random Forest)"""
        if model_name in self.trained_models:
            if hasattr(self.trained_models[model_name].named_steps[model_name], 'feature_importances_'):
                return self.trained_models[model_name].named_steps[model_name].feature_importances_
        return None
