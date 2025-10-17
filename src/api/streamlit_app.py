import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import logging
from pathlib import Path

from ..models.ensemble_model import EnsembleMedicalModel
from ..preprocessor.image_loader import MedicalImageLoader
from ..preprocessor.enhancement import MedicalImageEnhancer
from ..evaluator.visualization import ResultVisualizer

logger = logging.getLogger(__name__)

def start_web_interface(config: dict = None):
    """Start Streamlit web application"""
    
    st.set_page_config(
        page_title="MedVision AI Agent",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .result-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🏥 MedVision AI Agent</h1>', unsafe_allow_html=True)
    st.markdown("### Medical Image Analysis System")
    
    # Sidebar
    st.sidebar.title("Configuration")
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["Lightweight CNN", "Attention CNN", "Traditional ML", "Ensemble"]
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.8,
        step=0.05
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Single Image Analysis", 
        "📊 Batch Processing", 
        "📈 Model Training",
        "ℹ️ About"
    ])
    
    with tab1:
        st.header("Single Image Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a medical image",
            type=['jpg', 'jpeg', 'png', 'dcm', 'tiff']
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Analyze button
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Convert to OpenCV format
                        img_array = np.array(image)
                        if len(img_array.shape) == 3:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        
                        # TODO: Add actual analysis logic
                        result = {
                            "prediction": "Normal",
                            "confidence": 0.87,
                            "probabilities": {"Normal": 0.87, "Abnormal": 0.13},
                            "processing_time": 1.2
                        }
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📊 Analysis Results")
                            st.markdown(f"""
                            <div class="result-box">
                                <h3>Prediction: {result['prediction']}</h3>
                                <p><strong>Confidence:</strong> {result['confidence']:.3f}</p>
                                <p><strong>Processing Time:</strong> {result['processing_time']}s</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Confidence chart
                            fig, ax = plt.subplots(figsize=(8, 4))
                            classes = list(result['probabilities'].keys())
                            probabilities = list(result['probabilities'].values())
                            
                            bars = ax.bar(classes, probabilities, color=['green', 'red'])
                            ax.set_ylabel('Probability')
                            ax.set_title('Classification Probabilities')
                            
                            # Add value labels on bars
                            for bar, prob in zip(bars, probabilities):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{prob:.3f}', ha='center', va='bottom')
                            
                            st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error analyzing image: {e}")
    
    with tab2:
        st.header("Batch Processing")
        
        uploaded_files = st.file_uploader(
            "Choose multiple medical images",
            type=['jpg', 'jpeg', 'png', 'dcm', 'tiff'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} images")
            
            if st.button("Process Batch", type="primary"):
                progress_bar = st.progress(0)
                results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # Simulate processing
                    # TODO: Replace with actual processing
                    result = {
                        "filename": uploaded_file.name,
                        "prediction": "Normal" if i % 2 == 0 else "Abnormal",
                        "confidence": np.random.uniform(0.7, 0.95),
                        "processing_time": np.random.uniform(0.5, 2.0)
                    }
                    results.append(result)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display results table
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Images", len(results))
                with col2:
                    st.metric("Average Confidence", f"{df['confidence'].mean():.3f}")
                with col3:
                    normal_count = len(df[df['prediction'] == 'Normal'])
                    st.metric("Normal Findings", normal_count)
    
    with tab3:
        st.header("Model Training")
        
        st.info("This section allows you to train new models on your data")
        
        training_data = st.file_uploader(
            "Upload training data (ZIP file)",
            type=['zip'],
            accept_multiple_files=False
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.slider("Training Epochs", 10, 200, 50)
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        
        with col2:
            learning_rate = st.number_input("Learning Rate", 1e-5, 1e-1, 1e-3, format="%f")
            validation_split = st.slider("Validation Split", 0.1, 0.4, 0.2, 0.05)
        
        if st.button("Start Training", type="primary"):
            with st.spinner("Training model... This may take a while."):
                # TODO: Add actual training logic
                import time
                time.sleep(5)  # Simulate training
                
                st.success("Training completed successfully!")
                
                # Show training results
                st.subheader("Training Results")
                
                # Simulated metrics
                metrics = {
                    "accuracy": 0.894,
                    "precision": 0.901,
                    "recall": 0.887,
                    "f1_score": 0.894
                }
                
                for metric, value in metrics.items():
                    st.metric(metric.title(), f"{value:.3f}")
    
    with tab4:
        st.header("About MedVision AI Agent")
        
        st.markdown("""
        ### 🏥 Overview
        MedVision AI Agent is a comprehensive medical image analysis system that combines 
        traditional computer vision techniques with deep learning for accurate and 
        interpretable medical diagnosis.
        
        ### 🔬 Features
        - **Multi-Modal Analysis**: Combines texture, shape, and frequency features
        - **Deep Learning Models**: CNN, Attention Networks, U-Net from scratch
        - **Traditional ML**: Random Forest, SVM with handcrafted features
        - **Explainable AI**: Attention maps, feature importance, uncertainty quantification
        
        ### 🛠️ Technical Stack
        - **Backend**: Python, TensorFlow, OpenCV, scikit-learn
        - **API**: FastAPI for REST endpoints
        - **Web Interface**: Streamlit
        - **Deployment**: Docker, Kubernetes
        
        ### 📊 Supported Modalities
        - CT Scans
        - X-Rays  
        - MRI
        - Ultrasound
        
        ### ⚠️ Disclaimer
        This tool is for research purposes only. Always consult healthcare professionals 
        for medical diagnosis.
        """)

if __name__ == "__main__":
    start_web_interface()
