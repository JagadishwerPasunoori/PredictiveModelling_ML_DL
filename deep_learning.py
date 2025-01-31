import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_deep_learning_model(input_shape):
    """Build a deep learning model."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(1)  # Output layer for 1 target (math_score)
    ])

    model.compile(optimizer='adam', loss='mse')  # Use 'mse' as the loss function
    return model

def train_deep_learning_model(model, X_train, y_train, epochs=50, batch_size=32):
    """Train the deep learning model."""
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history