import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

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

def train_gradient_boosting(X_train, X_test, y_train):
    model = GradientBoostingClassifier(
        n_estimators=200, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(GradientBoostingClassifier(n_estimators=100, random_state=42))),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    return predictions

def save_predictions(test_ids, predictions, output_path):
    output_df = pd.DataFrame({
        'id': test_ids,
        'home_team_win': predictions
    })
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def main_gradient():
    train_path = 'train_data.csv'
    test_path = 'same_season_test_data.csv'
    output_path = 'predictions_gradient.csv'
    train_df, test_df = load_data(train_path, test_path)
    X_train, X_test, y_train, test_ids = preprocess_data(train_df, test_df)
    predictions = train_gradient_boosting(X_train, X_test, y_train)
    save_predictions(test_ids, predictions, output_path)

if __name__ == "__main__":
    main_gradient()
