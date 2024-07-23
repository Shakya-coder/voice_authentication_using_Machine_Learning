import os
import numpy as np
import librosa
import joblib
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer

def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def load_data(folder_path):
    feature_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            features = extract_features(os.path.join(folder_path, file_name))
            feature_list.append(features)
    return np.array(feature_list)

if __name__ == "__main__":
    folder_path = './training_data'
    X = load_data(folder_path)

    # Labels: 1 for in-class (user's voice)
    y = np.ones(X.shape[0])

    # Define the OneClassSVM model
    model = OneClassSVM()

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
        'nu': [0.1, 0.5, 0.9]
    }

    # Use GridSearchCV to find the best parameters
    grid_search = GridSearchCV(model, param_grid, scoring=make_scorer(accuracy_score, greater_is_better=True))
    grid_search.fit(X, y)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Print the best parameters and accuracy
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Training Accuracy: {grid_search.best_score_}")

    # Save the best model to a file
    model_filename = 'voice_authentication_model.pkl'
    joblib.dump(best_model, model_filename)
    print(f"Model saved as {model_filename}")
