
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical



# -----------------------------
# 1. Synthetic Complex Dataset Generation
# -----------------------------
def generate_complex_data(n_samples=3000, n_features=150, n_classes=4):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)

    W = np.random.randn(n_features, n_classes)
    logits = np.dot(X, W)
    y = np.argmax(logits, axis=1)

    df = pd.DataFrame(X)
    df["target"] = y
    return df


# -----------------------------
# 2. Data Loading & Saving
# -----------------------------
def create_and_save_dataset():
    df = generate_complex_data()
    df.to_csv("complex_feature_learning_data.csv", index=False)
    print("Dataset saved as complex_feature_learning_data.csv")


# -----------------------------
# 3. Data Loading & Inspection
# -----------------------------
def load_and_inspect_data():
    df = pd.read_csv("complex_feature_learning_data.csv")
    print("\nDataset Head:\n", df.head())
    print("\nDataset Info:\n", df.info())
    print("\nDataset Description:\n", df.describe())
    return df


# -----------------------------
# 4. Preprocessing
# -----------------------------
def preprocess_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "complex_scaler.pkl")
    print("Scaler saved as complex_scaler.pkl")

    return X_scaled, y


# -----------------------------
# 5. Deep Learning Model (Autoencoder + Classification Head)
# -----------------------------
def build_autoencoder(input_dim, enc_dim=64):
    inp = Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(inp)
    x = Dense(128, activation='relu')(x)
    encoded = Dense(enc_dim, activation='relu')(x)

    x = Dense(128, activation='relu')(encoded)
    x = Dense(256, activation='relu')(x)
    out = Dense(input_dim, activation='linear')(x)

    autoencoder = Model(inp, out)
    encoder = Model(inp, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


def build_classifier(input_dim, n_classes):
    inp = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inp)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    out = Dense(n_classes, activation='softmax')(x)

    model = Model(inp, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# -----------------------------
# 6. Training
# -----------------------------
def train_models(X_scaled, y):
    n_classes = len(np.unique(y))
    y_cat = to_categorical(y, num_classes=n_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_cat, test_size=0.2, random_state=42
    )

    autoencoder, encoder = build_autoencoder(X_scaled.shape[1])
    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Training Autoencoder for Automated Feature Learning...")
    autoencoder.fit(
        X_train, X_train,
        validation_data=(X_test, X_test),
        epochs=50,
        batch_size=32,
        callbacks=[early]
    )

    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)

    classifier = build_classifier(X_train_encoded.shape[1], n_classes)

    print("Training Classifier...")
    classifier.fit(
        X_train_encoded, y_train,
        validation_data=(X_test_encoded, y_test),
        epochs=40,
        batch_size=32,
        callbacks=[early]
    )

    encoder.save("complex_encoder.h5")
    classifier.save("complex_classifier.h5")

    print("Models Saved.")

    return classifier, encoder, X_test_encoded, y_test


# -----------------------------
# 7. Evaluation
# -----------------------------
def evaluate_model(classifier, X_test_encoded, y_test):
    y_pred = classifier.predict(X_test_encoded)
    y_true = np.argmax(y_test, axis=1)
    y_pred_class = np.argmax(y_pred, axis=1)

    print("\nAccuracy:", accuracy_score(y_true, y_pred_class))
    print("\nClassification Report:\n", classification_report(y_true, y_pred_class))

    cm = confusion_matrix(y_true, y_pred_class)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Deep Learning Feature Learning Confusion Matrix")
    plt.tight_layout()
    plt.savefig("deep_learning_feature_confusion_matrix.png")
    plt.show()


# -----------------------------
# 8. Inference Function
# -----------------------------
def feature_learning_predict(input_features):
    scaler = joblib.load("complex_scaler.pkl")
    encoder = load_model("complex_encoder.h5")
    classifier = load_model("complex_classifier.h5")

    input_features = np.array(input_features).reshape(1, -1)
    scaled = scaler.transform(input_features)

    encoded = encoder.predict(scaled)
    pred = classifier.predict(encoded)
    return np.argmax(pred)


# -----------------------------
# 9. Main Execution
# -----------------------------
def main():
    if not os.path.exists("complex_feature_learning_data.csv"):
        create_and_save_dataset()

    df = load_and_inspect_data()
    X_scaled, y = preprocess_data(df)
    classifier, encoder, X_test_encoded, y_test = train_models(X_scaled, y)
    evaluate_model(classifier, X_test_encoded, y_test)

    sample = np.random.randn(150)
    print("\nSample Prediction:", feature_learning_predict(sample))


if __name__ == "__main__":
    main()
