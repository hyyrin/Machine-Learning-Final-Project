import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

def load_data(train_path, test_path):
    """
    Load training and testing data from CSV files
    
    Args:
        train_path (str): Path to the training data CSV
        test_path (str): Path to the testing data CSV
    
    Returns:
        tuple: (train_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """
    Preprocess the training and testing data
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Testing dataframe
    
    Returns:
        tuple: (X_train, X_test, y_train, test_ids)
    """
    # Comprehensive feature selection
    features = [
        # Game context features
        'is_night_game', 'home_team_rest', 'away_team_rest', 
        'home_pitcher_rest', 'away_pitcher_rest', 'season',
        
        # Recent performance features (10-game rolling averages)
        'home_batting_batting_avg_10RA', 'home_batting_onbase_perc_10RA', 
        'home_batting_onbase_plus_slugging_10RA', 'home_batting_RBI_10RA',
        'away_batting_batting_avg_10RA', 'away_batting_onbase_perc_10RA', 
        'away_batting_onbase_plus_slugging_10RA', 'away_batting_RBI_10RA',
        
        # Pitching performance
        'home_pitching_earned_run_avg_10RA', 'home_pitching_SO_batters_faced_10RA',
        'home_pitching_H_batters_faced_10RA', 'home_pitching_BB_batters_faced_10RA',
        'away_pitching_earned_run_avg_10RA', 'away_pitching_SO_batters_faced_10RA',
        'away_pitching_H_batters_faced_10RA', 'away_pitching_BB_batters_faced_10RA',
        
        # Seasonal team performance
        'home_team_wins_mean', 'away_team_wins_mean',
        'home_team_spread_mean', 'away_team_spread_mean',
        'home_team_errors_mean', 'away_team_errors_mean',
        
        # Batting seasonal statistics
        'home_batting_leverage_index_avg_mean', 'away_batting_leverage_index_avg_mean',
        'home_batting_wpa_bat_mean', 'away_batting_wpa_bat_mean'
    ]
    
    # Prepare training data
    X_train = train_df[features].fillna(train_df[features].mean())
    y_train = train_df['home_team_win']
    
    # Prepare test data
    X_test = test_df[features].fillna(test_df[features].mean())
    test_ids = test_df['id']
    
    return X_train, X_test, y_train, test_ids

def create_model_pipeline(model):
    """
    Create a machine learning pipeline with feature selection and scaling
    
    Args:
        model: Sklearn classifier
    
    Returns:
        Pipeline: Configured machine learning pipeline
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
        ('classifier', model)
    ])

def train_and_predict_models(X_train, X_test, y_train):
    """
    Train multiple models and aggregate their predictions
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training target
    
    Returns:
        np.ndarray: Aggregated predictions
    """
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            min_samples_split=5, 
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42
        )
    }
    
    # Store individual model predictions
    all_predictions = []
    model_performances = {}
    
    for name, model in models.items():
        # Create pipeline
        pipeline = create_model_pipeline(model)
        
        # Perform cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
        
        # Train on full training data
        pipeline.fit(X_train, y_train)
        
        # Predict test data
        predictions = pipeline.predict(X_test)
        all_predictions.append(predictions)
        
        # Store model performance
        model_performances[name] = {
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std()
        }
    
    # Print model performances
    print("Model Performances:")
    for name, perf in model_performances.items():
        print(f"{name}:")
        print(f"  Cross-validation Accuracy: {perf['cv_mean_accuracy']:.4f} ± {perf['cv_std_accuracy']:.4f}")
    
    # Aggregate predictions (majority vote)
    final_predictions = np.round(np.mean(all_predictions, axis=0)).astype(bool)
    
    return final_predictions

def save_predictions(test_ids, predictions, output_path):
    """
    Save predictions to a CSV file
    
    Args:
        test_ids (pd.Series): Test set IDs
        predictions (np.ndarray): Predicted home team wins
        output_path (str): Path to save the output CSV
    """
    # Create prediction dataframe
    output_df = pd.DataFrame({
        'id': test_ids,
        'home_team_win': predictions
    })
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def main():
    # Paths for training, testing, and output files
    train_path = 'train_data.csv'  # Update with your training data path
    test_path = 'same_season_test_data.csv'    # Update with your testing data path
    output_path = 'predictions2.csv'
    
    # Load data
    train_df, test_df = load_data(train_path, test_path)
    
    # Preprocess data
    X_train, X_test, y_train, test_ids = preprocess_data(train_df, test_df)
    
    # Train models and make predictions
    predictions = train_and_predict_models(X_train, X_test, y_train)
    
    # Save predictions
    save_predictions(test_ids, predictions, output_path)

if __name__ == "__main__":
    main()