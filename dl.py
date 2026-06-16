# ----------------------------------------
# Deep Learning Model for Automated Feature Learning
# Breast Cancer Dataset
# ----------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


# ----------------------------------------
# 1. Load Dataset
# ----------------------------------------
def load_dataset():

    data = load_breast_cancer()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["diagnosis"] = data.target

    print("\nDataset Head\n")
    print(df.head())

    print("\nDataset Info\n")
    print(df.info())

    print("\nDataset Description\n")
    print(df.describe())

    return df


# ----------------------------------------
# 2. Data Preprocessing
# ----------------------------------------
def preprocess_data(df):

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "scaler.pkl")

    print("\nData Preprocessing Completed")

    return X_scaled, y


# ----------------------------------------
# 3. Build Autoencoder
# ----------------------------------------
def build_autoencoder(input_dim):

    input_layer = Input(shape=(input_dim,))

    x = Dense(128, activation="relu")(input_layer)
    x = Dense(64, activation="relu")(x)

    encoded = Dense(32, activation="relu")(x)

    x = Dense(64, activation="relu")(encoded)
    x = Dense(128, activation="relu")(x)

    decoded = Dense(input_dim, activation="linear")(x)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder, encoder


# ----------------------------------------
# 4. Build Classifier
# ----------------------------------------
def build_classifier(input_dim):

    input_layer = Input(shape=(input_dim,))

    x = Dense(32, activation="relu")(input_layer)
    x = Dropout(0.3)(x)

    x = Dense(16, activation="relu")(x)

    output = Dense(2, activation="softmax")(x)

    model = Model(input_layer, output)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ----------------------------------------
# 5. Train Models
# ----------------------------------------
def train_models(X_scaled, y):

    y_cat = to_categorical(y, 2)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_cat, test_size=0.2, random_state=42
    )

    autoencoder, encoder = build_autoencoder(X_scaled.shape[1])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    print("\nTraining Autoencoder...")

    autoencoder.fit(
        X_train,
        X_train,
        validation_data=(X_test, X_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)

    classifier = build_classifier(X_train_encoded.shape[1])

    print("\nTraining Classifier...")

    classifier.fit(
        X_train_encoded,
        y_train,
        validation_data=(X_test_encoded, y_test),
        epochs=40,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    encoder.save("encoder_model.keras")
    classifier.save("classifier_model.keras")

    print("\nModels Saved Successfully")

    return classifier, encoder, X_test_encoded, y_test


# ----------------------------------------
# 6. Evaluate Model
# ----------------------------------------
def evaluate_model(classifier, X_test_encoded, y_test):

    y_pred = classifier.predict(X_test_encoded)

    y_true = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_true, y_pred_classes)

    print("\nModel Accuracy:", acc)

    print("\nClassification Report\n")
    print(classification_report(y_true, y_pred_classes))

    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(6,5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig("confusion_matrix.png")

    plt.show()


# ----------------------------------------
# 7. Prediction Function
# ----------------------------------------
def predict_sample(input_features):

    scaler = joblib.load("scaler.pkl")

    encoder = load_model("encoder_model.keras")
    classifier = load_model("classifier_model.keras")

    input_features = np.array(input_features).reshape(1, -1)

    scaled = scaler.transform(input_features)

    encoded = encoder.predict(scaled)

    prediction = classifier.predict(encoded)

    return np.argmax(prediction)


# ----------------------------------------
# 8. Main
# ----------------------------------------
def main():

    df = load_dataset()

    X_scaled, y = preprocess_data(df)

    classifier, encoder, X_test_encoded, y_test = train_models(X_scaled, y)

    evaluate_model(classifier, X_test_encoded, y_test)

    sample = np.random.randn(X_scaled.shape[1])
    pred = predict_sample(sample)

    print("\nSample Prediction:", pred)


# ----------------------------------------
# Run
# ----------------------------------------
if __name__ == "__main__":
    main()