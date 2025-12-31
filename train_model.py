import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

print(">>> Script started")

# ==============================
# LOAD KEEL .dat DATASET
# ==============================
file_path = "dataset/NNGC1_dataset_E1_V1_001.dat"
print(">>> Dataset path:", file_path)
print(">>> File exists?", os.path.exists(file_path))

data_started = False
rows = []

try:
    with open(file_path, "r") as file:
        print(">>> File opened successfully")

        for i, line in enumerate(file):
            line = line.strip()

            # Debug first few lines
            if i < 5:
                print(f">>> Line {i}:", line)

            if line.lower() == "@data":
                data_started = True
                print(">>> @data section found")
                continue

            if data_started and line != "":
                rows.append(line.split(","))

    print(">>> Total data rows read:", len(rows))

except Exception as e:
    print("!!! ERROR opening or reading file:", e)
    exit()

if len(rows) == 0:
    print("!!! No data rows found after @data")
    exit()

columns = ["TimeStamp", "Lag04", "Lag03", "Lag02", "Lag01", "Class"]

try:
    df = pd.DataFrame(rows, columns=columns)
    df = df.astype(float)
    print(">>> DataFrame created successfully")
except Exception as e:
    print("!!! ERROR creating DataFrame or converting to float:", e)
    exit()

print(">>> Dataset loaded")
print(df.head())
print(">>> Shape:", df.shape)

# ==============================
# FEATURE & TARGET
# ==============================
print(">>> Preparing features and target")

try:
    X = df[["Lag01", "Lag02", "Lag03", "Lag04"]]
    y = df["Class"]
    print(">>> Features and target prepared")
except Exception as e:
    print("!!! ERROR preparing X or y:", e)
    exit()

# Time-series split (NO shuffle)
print(">>> Splitting train/test data")

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(">>> Train/test split done")
except Exception as e:
    print("!!! ERROR during train/test split:", e)
    exit()

# ==============================
# TRAIN REGRESSION MODEL
# ==============================
print(">>> Training regression model")

try:
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(">>> Model training completed")
except Exception as e:
    print("!!! ERROR during model training:", e)
    exit()

# ==============================
# MODEL EVALUATION
# ==============================
print(">>> Evaluating model")

try:
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    print(">>> MAE:", mae)
    print(">>> RMSE:", rmse)
except Exception as e:
    print("!!! ERROR during evaluation:", e)
    exit()

# ==============================
# SAVE TRAINED MODEL
# ==============================
print(">>> Saving model")

try:
    os.makedirs("model", exist_ok=True)

    with open("model/regression_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(">>> Model saved to model/regression_model.pkl")
except Exception as e:
    print("!!! ERROR saving model:", e)
    exit()

print(">>> Script finished successfully")
