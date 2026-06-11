import os
import sys
import logging
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score

sys.path.append('/app')
from shared.data_processor import StuntingDataProcessor
from shared.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class FederatedLearningClient:
    def __init__(self, client_name: str, data_path: str):
        self.client_name = client_name
        self.data_path = data_path
        # ADD CLASS WEIGHT TO HANDLE IMBALANCE!
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=config.RANDOM_STATE,
            class_weight='balanced',  # ← Kunci untuk imbalance data!
            solver='liblinear'  # Better for small data
        )
        self.processor = StuntingDataProcessor()
        self.X = None
        self.y = None
        self.num_samples = 0
        self.is_trained = False
        
        self.load_and_preprocess()
    
    def find_data_file(self):
        """Cari file data"""
        data_dir = '/app/data'
        possible_names = [
            f'{data_dir}/stunting_{self.client_name}.csv',
            f'{data_dir}/{self.client_name}.csv',
            f'{data_dir}/{self.client_name}.CSV',
        ]
        for path in possible_names:
            if os.path.exists(path):
                return path
        return None
    
    def load_and_preprocess(self):
        """Load data dari CSV"""
        logger.info(f"="*50)
        logger.info(f"LOADING DATA FOR {self.client_name.upper()}")
        logger.info(f"="*50)
        
        data_file = self.find_data_file()
        
        if data_file is None:
            logger.error(f"No data file found")
            self.X = np.random.rand(50, 11)
            self.y = np.random.randint(0, 2, 50)
            self.num_samples = 50
            return
        
        # Load CSV
        df = pd.read_csv(data_file, sep=";", encoding="latin1")
        logger.info(f"Raw data: {df.shape}")
        
        # Process
        self.X, self.y = self.processor.process(df, fit=True)
        self.num_samples = len(self.X)
        
        # Log class distribution
        n_stunting = (self.y == 1).sum()
        n_normal = (self.y == 0).sum()
        logger.info(f"✅ Data loaded:")
        logger.info(f"   Total: {self.num_samples} samples")
        logger.info(f"   Normal (0): {n_normal} ({n_normal/self.num_samples*100:.1f}%)")
        logger.info(f"   Stunting (1): {n_stunting} ({n_stunting/self.num_samples*100:.1f}%)")
        
        if n_stunting < n_normal * 0.3:
            logger.warning(f"⚠️  IMBALANCE DETECTED! Stunting only {n_stunting/self.num_samples*100:.1f}%")
            logger.warning(f"   Using class_weight='balanced' to handle imbalance")
    
    def get_parameters(self):
        if not self.is_trained:
            self.model.fit(self.X[:min(10, len(self.X))], self.y[:min(10, len(self.y))])
            self.is_trained = True
        return {
            'coef_': self.model.coef_.tolist(),
            'intercept_': self.model.intercept_.tolist()
        }
    
    def set_parameters(self, params):
        try:
            self.model.coef_ = np.array(params['coef_'])
            self.model.intercept_ = np.array(params['intercept_'])
            self.is_trained = True
        except Exception as e:
            logger.error(f"Error: {e}")
    
    def train_local(self, epochs=5):
        """Local training with metrics"""
        history = {'accuracy': [], 'f1': [], 'recall': []}
        
        if not self.is_trained:
            self.model.fit(self.X[:min(10, len(self.X))], self.y[:min(10, len(self.y))])
            self.is_trained = True
        
        for epoch in range(epochs):
            self.model.fit(self.X, self.y)
            
            # Calculate multiple metrics
            y_pred = self.model.predict(self.X)
            acc = (y_pred == self.y).mean()
            f1 = f1_score(self.y, y_pred, zero_division=0)
            recall = recall_score(self.y, y_pred, zero_division=0)
            
            history['accuracy'].append(acc)
            history['f1'].append(f1)
            history['recall'].append(recall)
            
            logger.debug(f"Epoch {epoch+1}: acc={acc:.3f}, f1={f1:.3f}, recall={recall:.3f}")
        
        # Log final metrics
        logger.info(f"Training done - Acc: {history['accuracy'][-1]:.3f}, F1: {history['f1'][-1]:.3f}")
        
        return self.get_parameters(), history
    
    def evaluate(self):
        """Evaluate with multiple metrics"""
        if not self.is_trained:
            return {'accuracy': 0.0, 'f1': 0.0, 'recall': 0.0}
        
        y_pred = self.model.predict(self.X)
        return {
            'accuracy': (y_pred == self.y).mean(),
            'f1': f1_score(self.y, y_pred, zero_division=0),
            'recall': recall_score(self.y, y_pred, zero_division=0),
            'samples': self.num_samples,
            'stunting_rate': self.y.mean()
        }

# Initialize
CLIENT_NAME = os.getenv('CLIENT_NAME', 'client')
DATA_PATH = f'/app/data/stunting_{CLIENT_NAME}.csv'
LOCAL_EPOCHS = getattr(config, 'LOCAL_EPOCHS', 5)

client = FederatedLearningClient(CLIENT_NAME, DATA_PATH)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'client': client.client_name,
        'samples': client.num_samples,
        'class_weight': 'balanced'
    })

@app.route('/api/client_info', methods=['GET'])
def client_info():
    return jsonify({
        'name': client.client_name,
        'num_samples': client.num_samples,
        'stunting_rate': float(client.y.mean()) if client.y is not None else 0
    })

@app.route('/api/update_model', methods=['POST'])
def update_model():
    try:
        data = request.json
        global_params = data.get('global_parameters')
        if global_params:
            client.set_parameters(global_params)
        
        local_params, history = client.train_local(epochs=LOCAL_EPOCHS)
        eval_metrics = client.evaluate()
        
        return jsonify({
            'success': True,
            'client_name': client.client_name,
            'parameters': local_params,
            'num_samples': client.num_samples,
            'accuracy': eval_metrics['accuracy'],
            'f1_score': eval_metrics['f1'],
            'recall': eval_metrics['recall'],
            'training_history': history
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/evaluate', methods=['GET'])
def evaluate():
    metrics = client.evaluate()
    return jsonify({
        'client_name': client.client_name,
        'accuracy': metrics['accuracy'],
        'f1_score': metrics['f1'],
        'recall': metrics['recall'],
        'num_samples': metrics['samples']
    })

if __name__ == '__main__':
    port = int(os.getenv('CLIENT_PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)