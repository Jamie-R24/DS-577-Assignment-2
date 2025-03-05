import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

"""
To test the accuracy and confidence of Week 6 Predictions, just add the Week6.mat file
to the folder "data". Running the program will spit out the test predictions in 
test_predictions.csv and will also run the K Means classifier. The validation_results.csv
shows the validation predictions as well as the actual value and confidence. 

elbow_method.png and kmeans_clusters.png displays the graph for the loss with the elbow method
and the graphical representation of the clusters. 
"""

# Function to load and preprocess data
def load_data(file_path):
    """
    Load .mat file and convert to pandas DataFrame with appropriate features
    """
    data = loadmat(file_path)
    # Extract the data array - assuming it's stored under a variable name in the .mat file
    # The actual variable name might differ - check the .mat file structure
    data_array = data['data'] if 'data' in data else next(value for key, value in data.items() 
                                                         if isinstance(value, np.ndarray) and value.ndim == 2)
    
    # Create DataFrame with appropriate column names
    df = pd.DataFrame(data_array, columns=['Timestamp', 'Latitude', 'Longitude', 'Accuracy', 'Label'])
    
    # Convert timestamp to datetime and extract features
    df['Datetime'] = df['Timestamp'].apply(lambda x: datetime.fromtimestamp(x))
    df['Hour'] = df['Datetime'].dt.hour
    df['Minute'] = df['Datetime'].dt.minute
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Round the label column to ensure it's an integer
    df['Label'] = df['Label'].round().astype(int)
    
    return df

# Function to extract features
def extract_features(df):
    """
    Extract relevant features for model training
    """
    # Create an explicit copy of the DataFrame slice
    features = df[['Latitude', 'Longitude', 'Accuracy', 'Hour', 'Minute', 'DayOfWeek', 'Weekend']].copy()
    
    # Create cyclical features for time - now using .loc to avoid warnings
    features.loc[:, 'Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
    features.loc[:, 'Hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)
    features.loc[:, 'Minute_sin'] = np.sin(2 * np.pi * df['Minute']/60)
    features.loc[:, 'Minute_cos'] = np.cos(2 * np.pi * df['Minute']/60)
    features.loc[:, 'DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek']/7)
    features.loc[:, 'DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek']/7)
    
    return features

# Function to train and evaluate the model
def train_model(X_train, y_train):
    """
    Train regression model with feature names handling using Pipeline and ColumnTransformer.
    """
    print("Training model...")

    # Define the preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', X_train.columns.tolist())  # Pass through all numerical columns
        ])

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Define hyperparameter grid
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    print(f"Best parameters: {grid_search.best_params_}")

    return best_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model, ensuring consistent preprocessing.
    """
    # Make predictions
    y_pred_raw = model.predict(X_test)

    # Round predictions to nearest integer
    y_pred = np.round(y_pred_raw).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate confidence for each prediction
    rf_regressor = model.named_steps['regressor']
    predictions = []
    for tree in rf_regressor.estimators_:
        # Preprocess X_test using the pipeline's preprocessor
        X_processed = model.named_steps['preprocessor'].transform(X_test)
        predictions.append(tree.predict(X_processed))

    predictions = np.array(predictions)
    confidence = 1 - np.std(predictions, axis=0) / (np.max(predictions, axis=0) - np.min(predictions, axis=0) + 1e-10)

    # Average confidence
    avg_confidence = np.mean(confidence)

    return accuracy, avg_confidence, y_pred, confidence

def kmeans_clustering(all_data):
    """
    Apply K-means clustering to find groups in location data (Longitude, Latitude).
    Uses the Elbow Method to determine the optimal number of clusters.
    """
    # Select only Longitude and Latitude columns
    X = all_data[['Longitude', 'Latitude']].to_numpy()

    # Elbow Method: Find the optimal number of clusters
    distortions = []
    K_range = range(1, 11)  # Checking K values from 1 to 10

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)  # Sum of squared distances to closest cluster center

    # Plot Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, distortions, marker='o', linestyle='-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Distortion (Inertia)')
    plt.title('Elbow Method for Optimal K')
    plt.grid()
    plt.savefig('elbow_method.png')

    optimal_k = 3  #After running once, 3 seemed to be the optimal value

    #K-means with the chosen K
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    all_data['Cluster'] = kmeans.fit_predict(X)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(all_data['Longitude'], all_data['Latitude'], c=all_data['Cluster'], cmap='viridis', alpha=0.6)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'K-means Clustering (K={optimal_k})')
    plt.legend()
    plt.savefig('kmeans_clusters.png')

    print(f"K-means clustering applied with {optimal_k} clusters.")


def main():
    
    print("Starting Regression location prediction...")
    
    # Load training data (weeks 1-4)
    training_data = []
    for week in range(1, 5):
        try:
            file_path = f"data/week{week}.mat"
            df = load_data(file_path)
            training_data.append(df)
            print(f"Loaded week {week} data: {len(df)} records")
        except Exception as e:
            print(f"Error loading week {week} data: {e}")
    
    # Combine 
    if training_data:
        train_df = pd.concat(training_data)
        print(f"Combined training data: {len(train_df)} records")
    else:
        print("No training data available. Exiting.")
        return
    
    # Extract
    X_train = extract_features(train_df)
    y_train = train_df['Label']
    
    # Train 
    model = train_model(X_train, y_train)
    
    # Load val data
    try:
        validation_df = load_data("data/week5.mat")
        print(f"Loaded validation data: {len(validation_df)} records")
        
       
        X_val = extract_features(validation_df)
        y_val = validation_df['Label']
      
        val_accuracy, val_confidence, val_predictions, val_pred_confidence = evaluate_model(model, X_val, y_val)
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Average prediction confidence: {val_confidence:.4f}")
    
        validation_results = pd.DataFrame({
            'Timestamp': validation_df['Timestamp'],
            'Actual_Label': y_val,
            'Predicted_Label': val_predictions,
            'Confidence': val_pred_confidence
        })
        validation_results.to_csv('validation_results.csv', index=False)
        
    except Exception as e:
        print(f"Error processing validation data: {e}")
    
    try:
        test_df = load_data("data/week6.mat")
        print(f"Loaded test data: {len(test_df)} records")

  
        X_test = extract_features(test_df)

        test_predictions_raw = model.predict(X_test)
        test_predictions = np.round(test_predictions_raw).astype(int)

        rf_regressor = model.named_steps['regressor'] # Get the regressor from the pipeline.
        predictions = []
        for tree in rf_regressor.estimators_:
            X_processed = model.named_steps['preprocessor'].transform(X_test)
            predictions.append(tree.predict(X_processed))
        predictions = np.array(predictions)
        test_confidence = 1 - np.std(predictions, axis=0) / (np.max(predictions, axis=0) - np.min(predictions, axis=0) + 1e-10)

        test_results = pd.DataFrame({
            'Timestamp': test_df['Timestamp'],
            'Predicted_Label': test_predictions,
            'Confidence': test_confidence
        })
        test_results.to_csv('test_predictions.csv', index=False)

        print(f"Test predictions saved to 'test_predictions.csv'")
        print(f"Average prediction confidence: {np.mean(test_confidence):.4f}")

        if 'Label' in test_df.columns:
            test_accuracy = accuracy_score(test_df['Label'], test_predictions)
            print(f"Test accuracy: {test_accuracy:.4f}")

    except Exception as e:
        print(f"No test data available or error: {e}")

    print("Linear Regression execution completed.")

    try:
        print("Starting K-means clustering...")

        all_weeks_data = []
        for week in range(1, 6):  # Load 5 weeks 
            try:
                file_path = f"data/week{week}.mat"
                df = load_data(file_path)
                all_weeks_data.append(df)
                print(f"Loaded week {week} data: {len(df)} records")
            except Exception as e:
                print(f"Error loading week {week} data: {e}")

        if all_weeks_data:
            all_data = pd.concat(all_weeks_data)
            print(f"Total data loaded: {len(all_data)} records")
        else:
            print("No data available. Exiting.")
            return

        # Apply K-means clustering
        kmeans_clustering(all_data)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()