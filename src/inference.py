import joblib
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report

def main():
    # Load saved model
    model = joblib.load("model_train.pkl")
    print("Model loaded successfully.")

    # Load digit dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    # Predict
    y_pred = model.predict(X)

    # Report
    print("Inference Results:")
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    main()
