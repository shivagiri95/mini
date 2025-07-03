import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# ------------------ Step 1: Load the NSL-KDD Dataset ------------------
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

data = pd.read_csv("/content/cic_dataset.csv", names=columns)

# Optional: Sample smaller subset
data = data.sample(n=100000, random_state=42).reset_index(drop=True)

# Drop non-feature column
data.drop('difficulty', axis=1, inplace=True)

# ------------------ Step 2: Data Preprocessing ------------------
# Convert numeric columns
for col in data.columns:
    if col not in ['protocol_type', 'service', 'flag', 'label']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
data.fillna(0, inplace=True)

# Encode categorical features
categorical_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}
for col in categorical_cols:
    data[col] = data[col].astype(str)
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Convert label to binary: 0 = normal, 1 = attack
data['label'] = data['label'].apply(lambda x: 0 if x == 'normal' else 1)

# ------------------ Step 3: Feature/Label Split ------------------
X = data.drop('label', axis=1).values
y = data['label'].astype(np.float32).values

# ------------------ Step 4: Normalize & Reshape ------------------
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))

# ------------------ Step 5: Train/Test Split ------------------
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_lstm, y, test_size=0.2, random_state=42
)

# ------------------ Step 6: Define & Train LSTM ------------------
model = Sequential()
model.add(LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary output

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train_lstm, y_train_lstm, epochs=16, batch_size=64, validation_split=0.2)

# ------------------ Step 7: Evaluate LSTM ------------------
loss, accuracy = model.evaluate(X_test_lstm, y_test_lstm)
print(f"\nLSTM Test Accuracy: {accuracy:.4f}")

# ------------------ Step 8: Plot Accuracy ------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('LSTM Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ------------------ Step 9: Classification Report & Confusion Matrix ------------------
y_pred_prob = model.predict(X_test_lstm).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)
y_true = y_test_lstm.astype(int)

print("\nLSTM Classification Report:")
print(classification_report(y_true, y_pred, labels=[0, 1], target_names=['normal', 'attack'], zero_division=0))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['normal', 'attack'], yticklabels=['normal', 'attack'])
plt.title("LSTM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------ Step 10: Classical Models ------------------
X_flat = X
y_flat = y.astype(int)

X_train_flat, X_test_flat, y_train_flat, y_test_flat = train_test_split(
    X_flat, y_flat, test_size=0.2, random_state=42
)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_flat, y_train_flat)
rf_preds = rf_model.predict(X_test_flat)
rf_acc = accuracy_score(y_test_flat, rf_preds)
rf_cm = confusion_matrix(y_test_flat, rf_preds)

# MLP
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, random_state=42)
mlp_model.fit(X_train_flat, y_train_flat)
mlp_preds = mlp_model.predict(X_test_flat)
mlp_acc = accuracy_score(y_test_flat, mlp_preds)
mlp_cm = confusion_matrix(y_test_flat, mlp_preds)

# ------------------ Step 11: Accuracy Comparison ------------------
models = ['LSTM', 'Random Forest', 'MLP']
accuracies = [accuracy, rf_acc, mlp_acc]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(True)
plt.show()

best_model_index = np.argmax(accuracies)
print(f"\nBest Model: {models[best_model_index]} with Accuracy: {accuracies[best_model_index]:.4f}")

best_cm = [cm, rf_cm, mlp_cm][best_model_index]
best_model_name = models[best_model_index]

plt.figure(figsize=(8, 6))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['normal', 'attack'], yticklabels=['normal', 'attack'])
plt.title(f'{best_model_name} Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ------------------ Step 12: Precision, Recall, F1 ------------------
def get_metrics(y_true, y_pred):
    return {
        "Precision": precision_score(y_true, y_pred, average='binary', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='binary', zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average='binary', zero_division=0)
    }

lstm_metrics = get_metrics(y_true, y_pred)
rf_metrics = get_metrics(y_test_flat, rf_preds)
mlp_metrics = get_metrics(y_test_flat, mlp_preds)

metrics_df = pd.DataFrame([lstm_metrics, rf_metrics, mlp_metrics], index=models)

metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title("Model Comparison: Precision, Recall, F1 Score")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(True)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

print("\nModel Performance Metrics:")
print(metrics_df.round(4))
