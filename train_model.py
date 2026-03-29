import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


MODEL_FILE = "models/model.pkl"
PIPELINE_FILE = "models/pipeline.pkl"


if not os.path.exists(MODEL_FILE):

    df = pd.read_csv("data/housing.csv")

    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    os.makedirs("input", exist_ok=True)
    X_test.to_csv("input/sample_input.csv", index=False)

    numerical_col = X_train.select_dtypes(include=["number"]).columns
    category_col = X_train.select_dtypes(include=["object"]).columns

    numerical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    category_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numerical_pipe, numerical_col),
            ("cat", category_pipe, category_col)
        ],
        verbose_feature_names_out=False
    )

    preprocess.set_output(transform="pandas")

    X_train_processed = preprocess.fit_transform(X_train)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_processed, y_train)

    X_test_processed = preprocess.transform(X_test)

    predictions = model.predict(X_test_processed)

    print("MAE:", mean_absolute_error(y_test, predictions))
    print("R2:", r2_score(y_test, predictions))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(preprocess, PIPELINE_FILE)

    print("Model trained and saved.")

else:

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    X_test = pd.read_csv("input/sample_input.csv")

    X_test_processed = pipeline.transform(X_test)

    predictions = model.predict(X_test_processed)

    X_test["prediction"] = predictions

    os.makedirs("output", exist_ok=True)
    X_test.to_csv("output/predictions.csv", index=False)

    print("Prediction completed and saved to output/predictions.csv")