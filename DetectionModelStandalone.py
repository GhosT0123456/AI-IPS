import pickle
import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

class DetectionModelStandalone:
    """Standalone detection model — works anywhere."""
    
    def __init__(self, model_path=None, pipeline_state_path=None):
        """Load model and preprocessing state."""
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "../../../models/UNSW_NB15_models/detection_model_catboost.cbm")
        if pipeline_state_path is None:
            pipeline_state_path = os.path.join(os.path.dirname(__file__), "../../../notebooks/UNSW_NB15_notebooks/pipeline_state.pkl")
        
        model_path = os.path.normpath(model_path)
        pipeline_state_path = os.path.normpath(pipeline_state_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(pipeline_state_path):
            raise FileNotFoundError(f"Pipeline state not found: {pipeline_state_path}")
        
        # Load model
        self.model = CatBoostClassifier()
        self.model.load_model(model_path, format='cbm')
        
        # Load pipeline state
        with open(pipeline_state_path, 'rb') as f:
            state = pickle.load(f)
        
        self.categorical_features = state['categorical_features']
        self.numerical_features = state['numerical_features']
        self.top_prop_categories = state['top_prop_categories']
        self.top_service_categories = state['top_service_categories']
        self.top_state_categories = state['top_state_categories']
        self.features_to_drop = state['features_to_drop']
    
    def preprocess(self, df):
        """Apply exact preprocessing."""
        df = df.copy()
        
        # 1. Basic cleaning
        columns_to_drop = ['swin', 'stcpb', 'dtcpb', 'dwin', 'attack_cat', 'response_body_len', 'src_ip', 'dst_ip']
        df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # 2. Handle outliers
        for col in self.numerical_features:
            if col in df.columns:
                lower = df[col].quantile(0.001)
                upper = df[col].quantile(0.999)
                df[col] = df[col].clip(lower, upper)
        
        # 3. Transform categories
        df['proto'] = np.where(df['proto'].isin(self.top_prop_categories), df['proto'], '-')
        df['service'] = np.where(df['service'].isin(self.top_service_categories), df['service'], '-')
        df['state'] = np.where(df['state'].isin(self.top_state_categories), df['state'], '-')
        
        # 4. Log1p features
        log_features = ['smean', 'dmean', 'sinpkt', 'dinpkt', 'sload', 'dload', 'sbytes', 'dbytes', 'sjit', 'djit']
        for feat in log_features:
            if feat in df.columns:
                df[feat] = np.log1p(df[feat])
        
        # 5. Drop correlated features
        df = df.drop(columns=self.features_to_drop, errors='ignore')
        
        # 6. Convert data types
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        for col in self.numerical_features:
            if col in df.columns and col not in self.features_to_drop:
                df[col] = df[col].astype('float32')
        
        return df
    
    def predict(self, df):
        """Predict on raw DataFrame."""
        df_processed = self.preprocess(df)
        pool = Pool(data=df_processed, cat_features=self.categorical_features)
        predictions = self.model.predict(pool)
        return predictions.tolist()
    
    def predict_proba(self, df):
        """Get prediction probabilities."""
        df_processed = self.preprocess(df)
        pool = Pool(data=df_processed, cat_features=self.categorical_features)
        return self.model.predict_proba(pool)