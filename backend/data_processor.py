import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model import preprocess_data, BidirectionalLSTMWithAttention
import joblib
import os

class SalesForecaster:
    """
    Handles data preprocessing, model training, and prediction for sales forecasting.
    """
    def __init__(self, sequence_length=30, prediction_horizon=10):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.features = None
        self.target_column = None
        
    def fit(self, data, target_column, features=None, lstm_units=64, dropout_rate=0.2,
            epochs=100, batch_size=32, validation_split=0.2):
        """
        Preprocess data and train the model.
        
        Args:
            data: DataFrame with time series data
            target_column: Column name of the target variable
            features: List of feature columns to use (if None, use all columns except target)
        """
        # Set columns
        self.target_column = target_column
        if features is None:
            self.features = [col for col in data.columns if col != target_column]
        else:
            self.features = features
            
        # Scale data
        data_scaled = data.copy()
        data_scaled[self.features] = self.feature_scaler.fit_transform(data[self.features])
        data_scaled[target_column] = self.target_scaler.fit_transform(data[[target_column]])
        
        # Preprocess data for LSTM
        X, y = preprocess_data(data_scaled, self.sequence_length, target_column, self.features)
        
        # Build and train model
        input_shape = (self.sequence_length, len(self.features))
        output_shape = 1  # Single step forecast
        
        self.model = BidirectionalLSTMWithAttention(
            input_shape=input_shape,
            output_shape=output_shape,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate
        )
        
        history = self.model.fit(
            X, y, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split
        )
        
        return history
    
    def predict(self, data, top_n=10, product_ids=None):
        """
        Generate forecasts for the next prediction_horizon steps.
        
        Args:
            data: DataFrame with recent time series data (at least sequence_length points)
            top_n: Number of top products to return
            product_ids: List of product IDs/names to associate with predictions
            
        Returns:
            DataFrame with forecasts for each product
        """
        if len(data) < self.sequence_length:
            raise ValueError(f"Input data must contain at least {self.sequence_length} time steps")
        
        # Scale input data
        data_scaled = data.copy()
        data_scaled[self.features] = self.feature_scaler.transform(data[self.features])
        
        # Prepare input sequence (most recent sequence_length points)
        X = np.array([data_scaled[self.features].values[-self.sequence_length:]])
        
        # Generate prediction
        y_pred_scaled = self.model.predict(X)
        
        # Inverse transform the prediction
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        
        # If product IDs are provided, create a DataFrame with results
        if product_ids is not None:
            results = pd.DataFrame({
                'product_id': product_ids,
                'forecast': y_pred.flatten()
            })
            results = results.sort_values('forecast', ascending=False).head(top_n)
            return results
        else:
            return y_pred
    
    def save(self, directory):
        """Save the model and scalers to the specified directory."""
        os.makedirs(directory, exist_ok=True)
        
        # Save model
        self.model.save(os.path.join(directory, 'bilstm_model.h5'))
        
        # Save scalers
        joblib.dump(self.feature_scaler, os.path.join(directory, 'feature_scaler.pkl'))
        joblib.dump(self.target_scaler, os.path.join(directory, 'target_scaler.pkl'))
        
        # Save parameters
        params = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'features': self.features,
            'target_column': self.target_column
        }
        joblib.dump(params, os.path.join(directory, 'params.pkl'))
    
    @classmethod
    def load(cls, directory):
        """Load a saved forecaster from the specified directory."""
        # Load parameters
        params = joblib.load(os.path.join(directory, 'params.pkl'))
        
        # Create instance
        forecaster = cls(
            sequence_length=params['sequence_length'],
            prediction_horizon=params['prediction_horizon']
        )
        forecaster.features = params['features']
        forecaster.target_column = params['target_column']
        
        # Load scalers
        forecaster.feature_scaler = joblib.load(os.path.join(directory, 'feature_scaler.pkl'))
        forecaster.target_scaler = joblib.load(os.path.join(directory, 'target_scaler.pkl'))
        
        # Load model
        forecaster.model = BidirectionalLSTMWithAttention.load(
            os.path.join(directory, 'bilstm_model.h5')
        )
        
        return forecaster
