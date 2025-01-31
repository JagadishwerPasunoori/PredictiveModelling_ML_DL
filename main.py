import joblib
import tensorflow as tf
from data_preprocessing import load_data, preprocess_data, split_data
from model_training import train_models, hyperparameter_tuning
from deep_learning import build_deep_learning_model, train_deep_learning_model
from evaluation import evaluate_model

def main():
    # Load and preprocess data
    data = load_data('data.csv')
    X, y, preprocessor = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train machine learning models
    best_model = train_models(X_train, y_train)
    best_model = hyperparameter_tuning(best_model, X_train, y_train)

    # Save the best model and preprocessor
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')

    # Evaluate the best machine learning model
    print("Evaluating Best Machine Learning Model...")
    evaluate_model(best_model, X_test, y_test)

    # Train and save the deep learning model
    input_shape = X_train.shape[1]
    dl_model = build_deep_learning_model(input_shape)
    train_deep_learning_model(dl_model, X_train, y_train, epochs=50, batch_size=32)
    dl_model.save('deep_learning_model.h5')  # Save the deep learning model

    # Evaluate the deep learning model
    print("Evaluating Deep Learning Model...")
    evaluate_model(dl_model, X_test, y_test)

if __name__ == "__main__":
    main()