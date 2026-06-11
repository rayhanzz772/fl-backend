import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)

class StuntingDataProcessor:
    """
    Preprocess data stunting untuk satu client (desa)
    Data tidak dicampur dengan client lain!
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False
        self.n_features = None
        
    def process(self, df, fit=True):
        """
        Process raw CSV menjadi features (X) dan target (y)
        
        Parameters:
        -----------
        df : DataFrame
            Raw data dari satu desa (Waru, Kemiri, atau Nangsri)
        fit : bool
            Fit preprocessing (True untuk training, False untuk inference)
        
        Returns:
        --------
        X : numpy array
            Features untuk training
        y : numpy array
            Target stunting (0=normal, 1=stunting)
        """
        
        logger.info("="*50)
        logger.info("STARTING DATA PREPROCESSING (LOCAL)")
        logger.info("="*50)
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # ==========================================
        # 1. CLEAN COLUMN NAMES
        # ==========================================
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace('ï»¿', '')
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.replace('/', '_')
        df.columns = df.columns.str.replace('-', '_')
        
        logger.info(f"Columns after cleaning: {df.columns.tolist()[:10]}...")
        
        # ==========================================
        # 2. CREATE TARGET STUNTING
        # ==========================================
        y = self._create_target(df, fit)
        
        # ==========================================
        # 3. SELECT FEATURES
        # ==========================================
        X = self._select_features(df)
        
        # ==========================================
        # 4. ENCODE CATEGORICAL FEATURES
        # ==========================================
        X = self._encode_categorical(X, fit)
        
        # ==========================================
        # 5. HANDLE MISSING VALUES
        # ==========================================
        X = self._handle_missing(X, fit)
        
        # ==========================================
        # 6. SCALE NUMERICAL FEATURES
        # ==========================================
        X = self._scale_features(X, fit)
        
        # ==========================================
        # 7. FINAL CHECK
        # ==========================================
        logger.info(f"Final X shape: {X.shape}")
        logger.info(f"Final y shape: {y.shape if y is not None else 'None'}")
        logger.info(f"Stunting rate: {y.mean()*100:.1f}%" if y is not None else "")
        
        return X, y
    
    def _create_target(self, df, fit):
        """Create stunting target from Z-Score"""
        # Cari kolom Z-score
        zscore_cols = ['ZS_BB_U', 'ZS_TB_U', 'ZS_BB_TB', 'ZS_TB_U', 'BB/U']
        available_zscore = [col for col in zscore_cols if col in df.columns]
        
        if len(available_zscore) == 0:
            logger.warning("No Z-Score column found! Using dummy target")
            return np.random.randint(0, 2, len(df))
        
        # Konversi ke numeric
        for col in available_zscore:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Stunting jika Z-Score < -2 SD
        conditions = []
        for col in available_zscore:
            conditions.append(df[col] < -2)
        
        if len(conditions) > 0:
            y = np.any(conditions, axis=0).astype(int)
            logger.info(f"Target created from: {available_zscore[0]} < -2")
            return y
        
        return np.zeros(len(df))
    
    def _select_features(self, df):
        """Select features for model training"""
        
        # Core features untuk stunting
        core_features = [
            'JK', 'BB_Lahir', 'TB_Lahir', 'Usia_Saat_Ukur',
            'Berat', 'Tinggi', 'Naik_Berat_Badan', 'Jml_Vit_A'
        ]
        
        # Additional features (jika ada)
        additional = ['KPSP', 'KIA', 'Kelas_Ibu_Balita']
        
        # Select available features
        selected = []
        for feat in core_features + additional:
            if feat in df.columns:
                selected.append(feat)
        
        logger.info(f"Selected {len(selected)} features: {selected}")
        
        # Handle missing features (fill with default)
        X = pd.DataFrame(index=df.index)
        for feat in selected:
            X[feat] = df[feat]
        
        self.n_features = len(selected)
        
        return X
    
    def _encode_categorical(self, X, fit):
        """Encode categorical features"""
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                # Handle NaN
                X[col] = X[col].fillna('missing')
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
                logger.debug(f"Encoded {col}: {len(le.classes_)} classes")
            else:
                le = self.label_encoders.get(col)
                if le:
                    X[col] = X[col].fillna('missing').astype(str)
                    # Handle unseen labels
                    X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    X[col] = le.transform(X[col])
        
        return X
    
    def _handle_missing(self, X, fit):
        """Handle missing values"""
        
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                # Numerical: fill with median
                if fit:
                    median_val = X[col].median()
                    setattr(self, f'median_{col}', median_val)
                else:
                    median_val = getattr(self, f'median_{col}', 0)
                X[col] = X[col].fillna(median_val)
            else:
                # Categorical: fill with mode
                if fit:
                    mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown'
                    setattr(self, f'mode_{col}', mode_val)
                else:
                    mode_val = getattr(self, f'mode_{col}', 'unknown')
                X[col] = X[col].fillna(mode_val)
        
        logger.info(f"Missing values handled: {X.isnull().sum().sum()} remaining")
        
        return X
    
    def _scale_features(self, X, fit):
        """Scale numerical features"""
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X)
        
        logger.info(f"Features scaled (mean=0, std=1)")
        
        return X_scaled
    
    def get_feature_names(self):
        """Get feature names"""
        if hasattr(self, 'feature_names'):
            return self.feature_names
        return [f'feature_{i}' for i in range(self.n_features)] if self.n_features else []
    
    def get_preprocessing_info(self):
        """Get preprocessing config for inference"""
        return {
            'n_features': self.n_features,
            'scaler_mean': self.scaler.mean_.tolist() if self.is_fitted else None,
            'scaler_scale': self.scaler.scale_.tolist() if self.is_fitted else None,
            'encoders': {k: list(v.classes_) for k, v in self.label_encoders.items()}
        }