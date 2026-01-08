import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================
# CONFIGURATION
# ==============================
# Set this to your dataset filename
DATASET_FILE = "dataset/NNGC1_dataset_F1_V1_006.dat"  # New larger dataset
# DATASET_FILE = "dataset/NNGC1_dataset_E1_V1_001.dat"  # Old dataset

print("="*60)
print("TRAFFIC VOLUME PREDICTION - MODEL TRAINING")
print("="*60)

# ==============================
# LOAD KEEL DATASET
# ==============================
rows = []
data_started = False

print(f"\nLoading dataset: {DATASET_FILE}")
try:
    with open(DATASET_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line.lower() == "@data":
                data_started = True
                continue
            if data_started and line != "":
                rows.append(line.split(","))
except FileNotFoundError:
    print(f"\n❌ ERROR: Dataset file not found: {DATASET_FILE}")
    print("Please ensure the dataset is in the correct location.")
    exit(1)

columns = ["TimeStamp", "Lag04", "Lag03", "Lag02", "Lag01", "Class"]
df = pd.DataFrame(rows, columns=columns).astype(float)

print(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# ==============================
# EXPLORATORY DATA ANALYSIS
# ==============================
print("\n" + "="*60)
print("DATASET STATISTICS")
print("="*60)

print("\nTarget Variable (Traffic Volume):")
print(f"  Mean:        {df['Class'].mean():.2f}")
print(f"  Std Dev:     {df['Class'].std():.2f}")
print(f"  Min:         {df['Class'].min():.2f}")
print(f"  Max:         {df['Class'].max():.2f}")
print(f"  Median:      {df['Class'].median():.2f}")
print(f"  Range:       {df['Class'].max() - df['Class'].min():.2f}")

# Determine if data is normalized or raw counts
data_range = df['Class'].max() - df['Class'].min()
if data_range < 200:
    print(f"\n📊 Data Type: NORMALIZED INDEX (0-100 scale)")
    data_type = "normalized"
elif data_range > 1000:
    print(f"\n📊 Data Type: RAW VEHICLE COUNTS (per hour)")
    data_type = "raw_counts"
else:
    print(f"\n📊 Data Type: UNKNOWN - Please verify units")
    data_type = "unknown"

print("\nLag Features Summary:")
for col in ["Lag01", "Lag02", "Lag03", "Lag04"]:
    print(f"  {col}: Mean={df[col].mean():.2f}, Std={df[col].std():.2f}, Range=[{df[col].min():.0f}, {df[col].max():.0f}]")

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# Check for outliers (values > 3 std devs from mean)
outliers = df['Class'][np.abs((df['Class'] - df['Class'].mean()) / df['Class'].std()) > 3]
print(f"Potential outliers (>3σ): {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

# ==============================
# PREPARE FEATURES & TARGET
# ==============================
print("\n" + "="*60)
print("PREPARING DATA FOR TRAINING")
print("="*60)

X = df[["Lag01", "Lag02", "Lag03", "Lag04"]]
y = df["Class"]

# Store statistics for DSS normalization
mean_volume = y.mean()
std_volume = y.std()
min_volume = y.min()
max_volume = y.max()
median_volume = y.median()

# Calculate percentiles for demand classification
percentiles = {
    'p10': np.percentile(y, 10),
    'p25': np.percentile(y, 25),
    'p50': np.percentile(y, 50),
    'p75': np.percentile(y, 75),
    'p90': np.percentile(y, 90)
}

print(f"\nFeature matrix: {X.shape}")
print(f"Target vector: {y.shape}")

print("\nTraffic Volume Percentiles:")
for p, val in percentiles.items():
    print(f"  {p}: {val:.2f}")

# ==============================
# TIME SERIES CROSS-VALIDATION
# ==============================
print("\n" + "="*60)
print("TIME SERIES CROSS-VALIDATION SETUP")
print("="*60)

# Use 80% for training, 20% for final testing
split_point = int(0.8 * len(df))
X_train_full = X[:split_point]
y_train_full = y[:split_point]
X_test = X[split_point:]
y_test = y[split_point:]

print(f"\nTraining set: {len(X_train_full)} samples ({len(X_train_full)/24:.1f} days)")
print(f"Test set:     {len(X_test)} samples ({len(X_test)/24:.1f} days)")
print(f"Train period: timestamps 0-{split_point-1}")
print(f"Test period:  timestamps {split_point}-{len(df)-1}")

# Time series cross-validation on training data
tscv = TimeSeriesSplit(n_splits=5)
print(f"\nTime Series CV: {tscv.n_splits} folds")

# ==============================
# TRAIN MODELS WITH CV
# ==============================
print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

models = {
    "lr": LinearRegression(),
    "rf": RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1),
    "gb": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
}

cv_scores = {name: [] for name in models.keys()}
final_models = {}
test_predictions = {}

os.makedirs("model", exist_ok=True)

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training: {name.upper()}")
    print(f"{'='*60}")
    
    # Cross-validation
    fold = 1
    for train_idx, val_idx in tscv.split(X_train_full):
        X_fold_train = X_train_full.iloc[train_idx]
        y_fold_train = y_train_full.iloc[train_idx]
        X_fold_val = X_train_full.iloc[val_idx]
        y_fold_val = y_train_full.iloc[val_idx]
        
        # Train on fold
        fold_model = type(model)(**model.get_params())
        fold_model.fit(X_fold_train, y_fold_train)
        
        # Validate
        val_pred = fold_model.predict(X_fold_val)
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, val_pred))
        fold_mae = mean_absolute_error(y_fold_val, val_pred)
        fold_mape = np.mean(np.abs((y_fold_val - val_pred) / y_fold_val)) * 100
        
        cv_scores[name].append(fold_rmse)
        
        print(f"  Fold {fold}: RMSE = {fold_rmse:.2f}, MAE = {fold_mae:.2f}, MAPE = {fold_mape:.2f}%")
        fold += 1
    
    # Print CV results
    mean_cv_rmse = np.mean(cv_scores[name])
    std_cv_rmse = np.std(cv_scores[name])
    print(f"\n  CV RMSE: {mean_cv_rmse:.2f} ± {std_cv_rmse:.2f}")
    
    # Train final model on full training set
    print(f"\n  Training final model on full training set...")
    model.fit(X_train_full, y_train_full)
    
    # Test set evaluation
    y_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    test_mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"  Test RMSE: {test_rmse:.2f}")
    print(f"  Test MAE:  {test_mae:.2f}")
    print(f"  Test MAPE: {test_mape:.2f}%")
    print(f"  Test R²:   {test_r2:.4f}")
    
    # Save model
    with open(f"model/{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    
    final_models[name] = {
        "model": model,
        "cv_rmse": mean_cv_rmse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_mape": test_mape,
        "test_r2": test_r2
    }
    test_predictions[name] = y_pred

# ==============================
# ENSEMBLE WEIGHTING
# ==============================
print("\n" + "="*60)
print("ENSEMBLE MODEL CONFIGURATION")
print("="*60)

# Use test RMSE for weighting
test_rmse_scores = {name: final_models[name]["test_rmse"] for name in models.keys()}

# Inverse RMSE weighting
inverse_rmse = {k: 1 / v for k, v in test_rmse_scores.items()}
total = sum(inverse_rmse.values())
weights = {k: v / total for k, v in inverse_rmse.items()}

print("\nTest RMSE Scores:")
for name, rmse in test_rmse_scores.items():
    print(f"  {name.upper()}: {rmse:.2f}")

print("\nEnsemble Weights (inverse RMSE):")
for name, weight in weights.items():
    print(f"  {name.upper()}: {weight:.4f} ({weight*100:.1f}%)")

# Ensemble predictions on test set
ensemble_pred = sum(weights[name] * test_predictions[name] 
                   for name in models.keys())

ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
ensemble_r2 = r2_score(y_test, ensemble_pred)
ensemble_mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100

print(f"\nEnsemble Performance:")
print(f"  Test RMSE: {ensemble_rmse:.2f}")
print(f"  Test MAE:  {ensemble_mae:.2f}")
print(f"  Test MAPE: {ensemble_mape:.2f}%")
print(f"  Test R²:   {ensemble_r2:.4f}")

# ==============================
# SAVE ENSEMBLE METADATA
# ==============================
ensemble_meta = {
    "rmse": test_rmse_scores,
    "weights": weights,
    "mean_volume": mean_volume,
    "std_volume": std_volume,
    "min_volume": min_volume,
    "max_volume": max_volume,
    "median_volume": median_volume,
    "percentiles": percentiles,
    "data_type": data_type,
    "dataset_samples": len(df),
    "ensemble_performance": {
        "rmse": ensemble_rmse,
        "mae": ensemble_mae,
        "mape": ensemble_mape,
        "r2": ensemble_r2
    },
    "individual_performance": {
        name: {
            "cv_rmse": final_models[name]["cv_rmse"],
            "test_rmse": final_models[name]["test_rmse"],
            "test_mae": final_models[name]["test_mae"],
            "test_mape": final_models[name]["test_mape"],
            "test_r2": final_models[name]["test_r2"]
        }
        for name in models.keys()
    }
}

with open("model/ensemble_meta.pkl", "wb") as f:
    pickle.dump(ensemble_meta, f)

print("\n✓ Ensemble metadata saved to model/ensemble_meta.pkl")

# ==============================
# VISUALIZATIONS
# ==============================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

os.makedirs("model/plots", exist_ok=True)

# 1. Model Comparison (4 subplots)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Test predictions comparison (first 200 points for clarity)
ax = axes[0, 0]
plot_samples = min(200, len(y_test))
test_indices = range(plot_samples)
ax.plot(test_indices, y_test.values[:plot_samples], 'k-', label='Actual', linewidth=2, alpha=0.8)
for name, pred in test_predictions.items():
    ax.plot(test_indices, pred[:plot_samples], '--', label=f'{name.upper()}', alpha=0.5, linewidth=1.5)
ax.plot(test_indices, ensemble_pred[:plot_samples], 'r-', label='Ensemble', linewidth=2.5, alpha=0.9)
ax.set_xlabel('Test Sample Index', fontsize=11)
ax.set_ylabel('Traffic Volume', fontsize=11)
ax.set_title('Test Set Predictions Comparison (First 200 samples)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Ensemble prediction vs actual (scatter)
ax = axes[0, 1]
# Sample points if too many
sample_size = min(1000, len(y_test))
sample_idx = np.random.choice(len(y_test), sample_size, replace=False)
ax.scatter(y_test.values[sample_idx], ensemble_pred[sample_idx], alpha=0.4, s=20, c='blue', edgecolors='none')
min_val = min(y_test.min(), ensemble_pred.min())
max_val = max(y_test.max(), ensemble_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Traffic Volume', fontsize=11)
ax.set_ylabel('Predicted Traffic Volume', fontsize=11)
ax.set_title(f'Ensemble Predictions (R² = {ensemble_r2:.3f}, MAPE = {ensemble_mape:.1f}%)', 
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Residuals
ax = axes[1, 0]
residuals = y_test.values - ensemble_pred
ax.scatter(ensemble_pred[sample_idx], residuals[sample_idx], alpha=0.4, s=20, c='green', edgecolors='none')
ax.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
ax.set_xlabel('Predicted Traffic Volume', fontsize=11)
ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
ax.set_title('Residual Plot (Ensemble)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# RMSE comparison
ax = axes[1, 1]
model_names = list(test_rmse_scores.keys()) + ['Ensemble']
rmse_values = list(test_rmse_scores.values()) + [ensemble_rmse]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax.bar(model_names, rmse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Test RMSE', fontsize=11)
ax.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, rmse_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('model/plots/model_evaluation.png', dpi=150, bbox_inches='tight')
print("✓ Saved: model/plots/model_evaluation.png")

# 2. Time series plot of full dataset
fig, ax = plt.subplots(figsize=(16, 5))
# Sample if too many points
if len(df) > 2000:
    sample_step = len(df) // 2000
    plot_df = df[::sample_step]
    ax.plot(plot_df['TimeStamp'], plot_df['Class'], linewidth=1, alpha=0.8)
else:
    ax.plot(df['TimeStamp'], df['Class'], linewidth=1.5, alpha=0.8)
ax.axvline(x=split_point, color='r', linestyle='--', linewidth=2.5, label='Train/Test Split', alpha=0.8)
ax.axhline(y=mean_volume, color='gray', linestyle=':', linewidth=2, label='Mean', alpha=0.6)
ax.set_xlabel('Time (hours)', fontsize=11)
ax.set_ylabel('Traffic Volume', fontsize=11)
ax.set_title(f'Complete Traffic Volume Time Series ({len(df)} samples)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model/plots/time_series_full.png', dpi=150, bbox_inches='tight')
print("✓ Saved: model/plots/time_series_full.png")

# 3. Distribution plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax1.hist(df['Class'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(mean_volume, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_volume:.1f}')
ax1.axvline(median_volume, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_volume:.1f}')
ax1.set_xlabel('Traffic Volume', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Traffic Volume Distribution', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Box plot
ax2.boxplot([df['Class']], labels=['Traffic Volume'], vert=True)
ax2.set_ylabel('Traffic Volume', fontsize=11)
ax2.set_title('Traffic Volume Box Plot', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model/plots/distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: model/plots/distribution.png")

# ==============================
# FINAL SUMMARY
# ==============================
print("\n" + "="*60)
print("TRAINING COMPLETE - SUMMARY")
print("="*60)

print(f"\n📊 Dataset Information:")
print(f"   • Type: {data_type.upper()}")
print(f"   • Samples: {len(df)} ({len(df)/24:.1f} days)")
print(f"   • Range: {min_volume:.0f} - {max_volume:.0f}")
print(f"   • Mean: {mean_volume:.1f} ± {std_volume:.1f}")

print(f"\n✓ Models trained and saved in 'model/' directory")
print(f"✓ Visualizations saved in 'model/plots/' directory")

print(f"\n📈 Model Performance:")
best_model = min(test_rmse_scores, key=test_rmse_scores.get)
print(f"   • Best Individual: {best_model.upper()} (RMSE: {test_rmse_scores[best_model]:.2f})")
print(f"   • Ensemble RMSE: {ensemble_rmse:.2f}")
print(f"   • Ensemble MAPE: {ensemble_mape:.2f}%")
print(f"   • Ensemble R²: {ensemble_r2:.4f}")

improvement = ((min(test_rmse_scores.values()) - ensemble_rmse) / min(test_rmse_scores.values())) * 100
if improvement > 0:
    print(f"   • Improvement: {improvement:.2f}% better than best individual")
else:
    print(f"   • Note: Ensemble slightly behind best model ({abs(improvement):.2f}%)")

print(f"\n🚀 Ready to run DSS application!")
print(f"   Run: streamlit run app.py")
print("="*60)