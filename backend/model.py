import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate, Dropout
from tensorflow.keras.layers import TimeDistributed, Attention, Layer

class AttentionLayer(Layer):
    """
    Attention layer for BiLSTM model.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # Alignment scores
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        
        # Calculate weights
        a = tf.keras.backend.softmax(e, axis=1)
        
        # Apply weights
        output = x * a
        
        return tf.keras.backend.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()

class BidirectionalLSTMWithAttention:
    """
    Bidirectional LSTM model with attention layer for time series forecasting.
    """
    def __init__(self, 
                 input_shape, 
                 output_shape, 
                 lstm_units=64, 
                 dropout_rate=0.2):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        
    def _build_model(self):
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Bidirectional LSTM layers
        lstm1 = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(inputs)
        dropout1 = Dropout(self.dropout_rate)(lstm1)
        
        lstm2 = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(dropout1)
        dropout2 = Dropout(self.dropout_rate)(lstm2)
        
        # Attention layer
        attention = AttentionLayer()(dropout2)
        
        # Output layer
        outputs = Dense(self.output_shape, activation='linear')(attention)
        
        # Build and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def fit(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        """Train the model on the given data."""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history
    
    def predict(self, X):
        """Generate predictions for the input data."""
        return self.model.predict(X)
    
    def save(self, filepath):
        """Save the model to disk."""
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load a saved model from disk."""
        model = tf.keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        return model

def preprocess_data(data, sequence_length, target_column, features=None):
    """
    Prepare data for the BiLSTM model.
    
    Args:
        data: DataFrame with time series data
        sequence_length: Number of time steps to use for each prediction
        target_column: Column name of the target variable
        features: List of feature columns to use (if None, use all columns except target)
    
    Returns:
        X: Input sequences
        y: Target values
    """
    if features is None:
        features = [col for col in data.columns if col != target_column]
    
    X = []
    y = []
    
    for i in range(len(data) - sequence_length):
        X.append(data[features].values[i:i+sequence_length])
        y.append(data[target_column].values[i+sequence_length])
    
    return np.array(X), np.array(y)
