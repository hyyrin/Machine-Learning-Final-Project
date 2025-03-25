import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
    # Select relevant features
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
    
    # Prepare training data
    X_train = train_df[features].fillna(train_df[features].mean())
    y_train = train_df['home_team_win']
    
    # Prepare test data
    X_test = test_df[features].fillna(test_df[features].mean())
    test_ids = test_df['id']
    
    return X_train, X_test, y_train, test_ids

def train_and_predict_model(X_train, X_test, y_train):
    """
    Train the model and make predictions
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training target
    
    Returns:
        np.ndarray: Predicted home team win probabilities
    """
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Predict probabilities and convert to boolean
    y_pred_proba = model.predict_proba(X_test_scaled)
    y_pred = (y_pred_proba[:, 1] >= 0.5)
    
    return y_pred

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
    test_path = '2024_test_data.csv'    # Update with your testing data path
    output_path = 'predictions_logistic.csv'
    
    # Load data
    train_df, test_df = load_data(train_path, test_path)
    
    # Preprocess data
    X_train, X_test, y_train, test_ids = preprocess_data(train_df, test_df)
    
    # Train model and make predictions
    predictions = train_and_predict_model(X_train, X_test, y_train)
    
    # Save predictions
    save_predictions(test_ids, predictions, output_path)
    
    # Optional: Print model performance on training data
    print("\nModel Performance:")
    X_train_scaled = StandardScaler().fit_transform(X_train)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred_train = model.predict(X_train_scaled)
    
    print(f"Training Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_train, y_pred_train))

if __name__ == "__main__":
    main()
