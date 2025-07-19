import json
import joblib
import os
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load hyperparameters from JSON config file
def load_config(config_path="config/config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

# Train model using Logistic Regression
def train_model(X, y, config):
    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X, y)
    return model

def main():
    # Load dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    # Load hyperparameters
    config = load_config()

    # Train model
    model = train_model(X, y, config)

    # Evaluate (optional)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Training accuracy: {acc:.4f}")

    # Save model
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "model_train.pkl")
    print("Model saved to model_train.pkl")

if __name__ == "__main__":
    main()
