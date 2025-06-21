import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(train_df, test_df):
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
        'home_batting_wpa_bat_mean', 'away_batting_wpa_bat_mean',
                # new
        'away_batting_wpa_bat_std', 'away_batting_wpa_bat_skew',
        'away_batting_RBI_mean', 'away_batting_RBI_std', 'away_batting_RBI_skew',
        'home_pitching_earned_run_avg_mean', 'home_pitching_earned_run_avg_std',
        'home_pitching_earned_run_avg_skew', 'home_pitching_SO_batters_faced_mean',
        'home_pitching_SO_batters_faced_std', 'home_pitching_SO_batters_faced_skew',
        'home_pitching_H_batters_faced_mean', 'home_pitching_H_batters_faced_std',
        'home_pitching_H_batters_faced_skew', 'home_pitching_BB_batters_faced_mean',
        'home_pitching_BB_batters_faced_std', 'home_pitching_BB_batters_faced_skew',
        'home_pitching_leverage_index_avg_mean', 'home_pitching_leverage_index_avg_std',
        'home_pitching_leverage_index_avg_skew', 'home_pitching_wpa_def_mean',
        'home_pitching_wpa_def_std', 'home_pitching_wpa_def_skew',
        'away_pitching_earned_run_avg_mean', 'away_pitching_earned_run_avg_std',
        'away_pitching_earned_run_avg_skew', 'away_pitching_SO_batters_faced_mean',
        'away_pitching_SO_batters_faced_std', 'away_pitching_SO_batters_faced_skew',
        'away_pitching_H_batters_faced_mean', 'away_pitching_H_batters_faced_std',
        'away_pitching_H_batters_faced_skew', 'away_pitching_BB_batters_faced_mean',
        'away_pitching_BB_batters_faced_std', 'away_pitching_BB_batters_faced_skew',
        'away_pitching_leverage_index_avg_mean', 'away_pitching_leverage_index_avg_std',
        'away_pitching_leverage_index_avg_skew', 'away_pitching_wpa_def_mean',
        'away_pitching_wpa_def_std', 'away_pitching_wpa_def_skew',
        'home_pitcher_earned_run_avg_mean', 'home_pitcher_earned_run_avg_std',
        'home_pitcher_earned_run_avg_skew', 'home_pitcher_SO_batters_faced_mean',
        'home_pitcher_SO_batters_faced_std', 'home_pitcher_SO_batters_faced_skew',
        'home_pitcher_H_batters_faced_mean', 'home_pitcher_H_batters_faced_std',
        'home_pitcher_H_batters_faced_skew', 'home_pitcher_BB_batters_faced_mean',
        'home_pitcher_BB_batters_faced_std', 'home_pitcher_BB_batters_faced_skew',
        'home_pitcher_leverage_index_avg_mean', 'home_pitcher_leverage_index_avg_std',
        'home_pitcher_leverage_index_avg_skew', 'home_pitcher_wpa_def_mean',
        'home_pitcher_wpa_def_std', 'home_pitcher_wpa_def_skew',
        'away_pitcher_earned_run_avg_mean', 'away_pitcher_earned_run_avg_std',
        'away_pitcher_earned_run_avg_skew', 'away_pitcher_SO_batters_faced_mean',
        'away_pitcher_SO_batters_faced_std', 'away_pitcher_SO_batters_faced_skew',
        'away_pitcher_H_batters_faced_mean', 'away_pitcher_H_batters_faced_std',
        'away_pitcher_H_batters_faced_skew', 'away_pitcher_BB_batters_faced_mean',
         'away_pitcher_BB_batters_faced_std', 'away_pitcher_BB_batters_faced_skew',
        'away_pitcher_leverage_index_avg_mean', 'away_pitcher_leverage_index_avg_std',
        'away_pitcher_leverage_index_avg_skew', 'away_pitcher_wpa_def_mean',
        'away_pitcher_wpa_def_std', 'away_pitcher_wpa_def_skew'
      
    ]
    X_train = train_df[features].fillna(train_df[features].mean())
    y_train = train_df['home_team_win']
    X_test = test_df[features].fillna(test_df[features].mean())
    test_ids = test_df['id']
    return X_train, X_test, y_train, test_ids

def train_random_forest(X_train, X_test, y_train, y_test=None):
    """
    Train a Random Forest model and optionally evaluate accuracy on a test set.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series, optional): Test labels for accuracy evaluation.
    
    Returns:
        tuple: (predictions, accuracy) if y_test is provided, otherwise only predictions.
    """
    model = RandomForestClassifier(
        n_estimators=300, 
        max_depth=5, 
        min_samples_split=3, 
        random_state=42
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    # Calculate accuracy if y_test is provided
    accuracy = None
    if y_test is not None:
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy:.4f}")

    return predictions, accuracy

def save_predictions(test_ids, predictions, output_path):
    output_df = pd.DataFrame({
        'id': test_ids,
        'home_team_win': predictions
    })
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")



def main_random_forest():
    # File paths
    train_path = 'train_data.csv'
    test_path = 'same_season_test_data.csv'
    output_path = 'predictions_forest_v2.csv'
    
    # Load and preprocess data
    train_df, test_df = load_data(train_path, test_path)
    X_train, X_test, y_train, test_ids = preprocess_data(train_df, test_df)

    # Split train data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Train model and calculate accuracy
    print("Training Random Forest model...")
    predictions, accuracy = train_random_forest(X_train_split, X_val_split, y_train_split, y_val_split)

    # Save predictions for test set
    test_predictions, _ = train_random_forest(X_train, X_test, y_train)
    save_predictions(test_ids, test_predictions, output_path)

    
if __name__ == "__main__":
    main_random_forest()
