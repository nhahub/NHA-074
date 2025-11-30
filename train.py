import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import recall_score
from joblib import dump
import mlflow
import mlflow.sklearn

# ==================== Load dataset ====================
df = pd.read_csv("notebook/Lung Cancer Dataset.csv")
df['PULMONARY_DISEASE'] = df['PULMONARY_DISEASE'].map({'YES': 1, 'NO': 0})

selected_features = [
    'SMOKING',
    'SMOKING_FAMILY_HISTORY',
    'THROAT_DISCOMFORT',
    'BREATHING_ISSUE',
    'STRESS_IMMUNE',
    'ENERGY_LEVEL',
    'IMMUNE_WEAKNESS',
    'FAMILY_HISTORY'
]

X = df[selected_features]
y = df['PULMONARY_DISEASE']

# ==================== Split dataset ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==================== Scale features ====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features)

# ==================== Hyperparameter search ====================
param_dist = {
    'n_estimators': randint(10, 200),
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 11),
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='recall',
    n_jobs=-1
)

# ==================== MLflow Experiment ====================
mlflow.set_experiment("Lung_Cancer_Experiments")  # تنظيم التجارب

with mlflow.start_run(run_name="Patient_History_Training"):

    # ====================  params  ====================
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("selected_features", selected_features)

    # ==================== Fit RandomizedSearch ====================
    random_search.fit(X_train_scaled, y_train)
    best_model = random_search.best_estimator_

    # ====================  hyperparameters ====================
    mlflow.log_params(best_model.get_params())

    # ==================== Evaluate ====================
    y_pred = best_model.predict(X_test_scaled)
    rec_test = recall_score(y_test, y_pred)
    print("Recall:", rec_test)
    mlflow.log_metric("recall", rec_test)

    # ==================== Save model locally ====================
    dump(best_model, "models/patient_history.joblib")
    dump(scaler, "models/scaler.joblib")

    # ==================== Log model to MLflow ====================
    mlflow.sklearn.log_model(
        best_model,
        "patient_history_model",
        registered_model_name="LungCancerModel"  # 
    )

print("Training completed & MLflow logging done")
