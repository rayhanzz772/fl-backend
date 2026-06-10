import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import logging
from typing import Tuple, Optional, Dict
from .config import config

logger = logging.getLogger(__name__)

class StuntingDataProcessor:
    """
    Complete data processor for stunting dataset
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_fitted = False
        
    def process(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Main processing pipeline
        """
        logger.info("="*60)
        logger.info("STARTING DATA PREPROCESSING")
        logger.info(f"Initial shape: {df.shape}")
        logger.info("="*60)
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # STEP 1: Drop identitas columns
        drop_cols = [c for c in config.DROP_COLUMNS if c in df.columns]
        df = df.drop(columns=drop_cols, errors='ignore')
        logger.info(f"Step 1 - Dropped {len(drop_cols)} ID columns: {drop_cols}")
        logger.info(f"  Remaining columns: {df.shape[1]}")
        
        # STEP 2: Create age in months from birth date
        if 'Tgl Lahir' in df.columns:
            df['Tgl Lahir'] = pd.to_datetime(df['Tgl Lahir'], errors='coerce')
            today = pd.Timestamp.today()
            df['umur_bulan'] = ((today - df['Tgl Lahir']).dt.days / 30.44).round()
            df = df.drop(columns=['Tgl Lahir', 'Usia Saat Ukur'], errors='ignore')
            logger.info(f"Step 2 - Created 'umur_bulan' from birth date")
        
        # STEP 3: Handle geographic data
        geo_cols = ['Prov', 'Kab/Kota', 'Kec', 'Desa/Kel']
        for col in geo_cols:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le:
                        df[col] = df[col].astype(str).apply(
                            lambda x: x if x in le.classes_ else 'unknown'
                        )
                        df[col] = le.transform(df[col])
        logger.info(f"Step 3 - Encoded geographic columns: {geo_cols}")
        
        # STEP 4: Drop redundant columns
        redundant_cols = ['BB/U', 'ZS BB/U', 'TB/U']
        df = df.drop(columns=[c for c in redundant_cols if c in df.columns], errors='ignore')
        logger.info(f"Step 4 - Dropped redundant columns: {redundant_cols}")
        
        # STEP 5: Select core features
        available_features = [c for c in config.CORE_FEATURES if c in df.columns]
        available_features.extend([c for c in geo_cols if c in df.columns])
        self.feature_columns = available_features
        
        logger.info(f"Step 5 - Selected {len(available_features)} features:")
        logger.info(f"  {available_features}")
        
        X = df[available_features].copy()
        
        # STEP 6: Handle missing values
        # Numerical columns - fill with median
        num_cols = X.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if fit:
                median_val = X[col].median()
                setattr(self, f'{col}_median', median_val)
            else:
                median_val = getattr(self, f'{col}_median', 0)
            X[col] = X[col].fillna(median_val)
        
        # Categorical columns - fill with mode
        cat_cols = X.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if fit:
                mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown'
                setattr(self, f'{col}_mode', mode_val)
            else:
                mode_val = getattr(self, f'{col}_mode', 'unknown')
            X[col] = X[col].fillna(mode_val)
        
        missing_after = X.isnull().sum().sum()
        logger.info(f"Step 6 - Missing values handled: {missing_after} remaining")
        
        # STEP 7: Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    X[col] = X[col].astype(str).apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    X[col] = le.transform(X[col])
        logger.info(f"Step 7 - Encoded {len(categorical_cols)} categorical columns")
        
        # STEP 8: Handle outliers using IQR
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fit:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                setattr(self, f'{col}_lower', lower_bound)
                setattr(self, f'{col}_upper', upper_bound)
            else:
                lower_bound = getattr(self, f'{col}_lower', X[col].min())
                upper_bound = getattr(self, f'{col}_upper', X[col].max())
            X[col] = X[col].clip(lower_bound, upper_bound)
        logger.info(f"Step 8 - Handled outliers for {len(numeric_cols)} numeric columns")
        
        # STEP 9: Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X)
        logger.info(f"Step 9 - Features scaled (mean=0, std=1)")
        
        # STEP 10: Create target variable
        y = None
        if config.TARGET_COLUMN in df.columns:
            zs = pd.to_numeric(df[config.TARGET_COLUMN], errors='coerce')
            
            # Handle missing ZS
            if fit:
                zs_median = zs.median()
                setattr(self, 'zs_median', zs_median)
            else:
                zs_median = getattr(self, 'zs_median', 0)
            
            zs = zs.fillna(zs_median)
            y = (zs < config.STUNTING_THRESHOLD).astype(int)
            
            logger.info(f"Step 10 - Created target 'stunting'")
            logger.info(f"  Stunting rate: {y.mean()*100:.2f}%")
            logger.info(f"  Stunting (1): {(y==1).sum()} samples")
            logger.info(f"  Normal (0): {(y==0).sum()} samples")
        
        # Final summary
        logger.info("="*60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info(f"Final features shape: {X_scaled.shape}")
        logger.info(f"Features: {self.feature_columns}")
        logger.info("="*60)
        
        return X_scaled, y
    
    def get_feature_names(self) -> list:
        return self.feature_columns
    
    def get_preprocessing_info(self) -> dict:
        return {
            'features': self.feature_columns,
            'n_features': len(self.feature_columns),
            'scaler_mean': self.scaler.mean_.tolist() if self.is_fitted else None,
            'scaler_scale': self.scaler.scale_.tolist() if self.is_fitted else None,
            'encoders': {k: list(v.classes_) for k, v in self.label_encoders.items()}
        }


# Column configuration for reference
COLUMN_CONFIG = {
    'dropped': ['No', 'NIK', 'Nama', 'Nama Ortu', 'Alamat', 'Detail', 'RW', 'RT', 'Posyandu'],
    'redundant': ['BB/U', 'ZS BB/U', 'TB/U', 'Usia Saat Ukur', 'Tanggal Pengukuran', 'Cara Ukur', 'LiLA'],
    'core_features': config.CORE_FEATURES,
    'geographic': ['Prov', 'Kab/Kota', 'Kec', 'Desa/Kel'],
    'target': config.TARGET_COLUMN
}