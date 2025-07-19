import os
import json
import pytest
from sklearn.linear_model import LogisticRegression
from src.train import load_config, train_model
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

CONFIG_PATH = "config/config.json"

def test_config_loads():
    config = load_config(CONFIG_PATH)
    assert "C" in config
    assert "solver" in config
    assert "max_iter" in config
    assert isinstance(config["C"], float)
    assert isinstance(config["solver"], str)
    assert isinstance(config["max_iter"], int)

def test_model_type():
    digits = load_digits()
    X, y = digits.data, digits.target
    config = load_config(CONFIG_PATH)
    model = train_model(X, y, config)
    assert isinstance(model, LogisticRegression)

def test_model_accuracy():
    digits = load_digits()
    X, y = digits.data, digits.target
    config = load_config(CONFIG_PATH)
    model = train_model(X, y, config)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    assert accuracy > 0.85
