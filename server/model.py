import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class FederatedModel:
    def __init__(self, n_features=None):
        self.n_features = n_features
        self.model = LogisticRegression(
            max_iter=1000, 
            random_state=42, 
            C=1.0,
            class_weight='balanced'
        )
        
        if n_features:
            # Initialize with dummy data
            dummy_X = np.random.rand(10, n_features)
            dummy_y = np.random.randint(0, 2, 10)
            self.model.fit(dummy_X, dummy_y)
            logger.info(f"Model initialized with {n_features} features")
    
    def get_parameters(self):
        """Get model parameters"""
        return {
            'coef_': self.model.coef_.tolist(),
            'intercept_': self.model.intercept_.tolist(),
            'n_features': self.model.coef_.shape[1]
        }
    
    def set_parameters(self, params):
        """Set model parameters"""
        try:
            self.model.coef_ = np.array(params['coef_'])
            self.model.intercept_ = np.array(params['intercept_'])
            self.n_features = params['n_features']
            logger.info("Model parameters updated successfully")
        except Exception as e:
            logger.error(f"Error setting parameters: {str(e)}")
            raise
    
    def aggregate_fedavg(self, client_params, client_samples):
        """Federated Averaging aggregation"""
        total_samples = sum(client_samples)
        if not client_params:
            raise ValueError("No client parameters provided for aggregation")
        
        # Infer the expected shape from the first client update instead of the
        # server's initial placeholder model shape.
        target_coef = np.asarray(client_params[0]['coef_'], dtype=float)
        if target_coef.ndim == 1:
            target_coef = target_coef.reshape(1, -1)
        current_shape = target_coef.shape
        
        # Weighted average for coefficients
        aggregated_coef = np.zeros(current_shape)
        aggregated_intercept = np.zeros_like(np.asarray(client_params[0]['intercept_'], dtype=float))
        
        for index, (params, num_samples) in enumerate(zip(client_params, client_samples)):
            weight = num_samples / total_samples
            client_coef = np.asarray(params['coef_'], dtype=float)
            
            # Ensure shape compatibility
            if client_coef.shape != current_shape:
                try:
                    client_coef = client_coef.reshape(current_shape)
                except ValueError as exc:
                    raise ValueError(
                        f"Client {index} coefficient shape {client_coef.shape} is incompatible with expected shape {current_shape}"
                    ) from exc
            
            aggregated_coef += client_coef * weight
            aggregated_intercept += np.asarray(params['intercept_'], dtype=float) * weight
        
        # Update model
        self.model.coef_ = aggregated_coef
        self.model.intercept_ = aggregated_intercept
        self.model.n_features_in_ = current_shape[1]
        self.n_features = current_shape[1]
        
        logger.info(f"Aggregated {len(client_params)} clients with {total_samples} total samples")
        
        return self.get_parameters()
    
    def save(self, path):
        """Save model to file"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, path):
        """Load model from file"""
        if os.path.exists(path):
            self.model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
            return True
        return False
    
    def predict(self, X):
        """Make prediction"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)