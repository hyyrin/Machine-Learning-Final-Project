import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(train_df, test_df):
    features = [
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

def create_stacking_model():
    """
    Create a stacking classifier with advanced base learners and a meta-learner.
    """
    # Base learners
    base_learners = [
        ('random_forest', RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)),
        ('gradient_boosting', GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))
    ]
    
    # Meta-learner
    meta_learner = LogisticRegression(max_iter=1000)
    
    # Stacking Classifier
    stacking_model = StackingClassifier(
        estimators=base_learners, 
        final_estimator=meta_learner, 
        cv=5
    )
    return stacking_model

def train_and_predict(X_train, X_test, y_train):
    """
    Train the stacking model and make predictions.
    """
    # Preprocessing pipeline with feature selection
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
        ('stacking', create_stacking_model())
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Predict test set
    predictions = pipeline.predict(X_test)
    return predictions

def save_predictions(test_ids, predictions, output_path):
    """
    Save predictions to a CSV file.
    """
    output_df = pd.DataFrame({
        'id': test_ids,
        'home_team_win': predictions
    })
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def main():
    train_path = 'train_data.csv'  # Update with your training data path
    test_path = '2024_test_data.csv'    # Update with your testing data path
    output_path = 'predictions_stack.csv'
    
    # Load and preprocess data
    train_df, test_df = load_data(train_path, test_path)
    X_train, X_test, y_train, test_ids = preprocess_data(train_df, test_df)
    
    # Train and predict
    predictions = train_and_predict(X_train, X_test, y_train)
    
    # Save predictions
    save_predictions(test_ids, predictions, output_path)

if __name__ == "__main__":
    main()
