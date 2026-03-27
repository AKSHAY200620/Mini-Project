import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import shap
import pickle

#load
data = pd.read_excel(r'C:\Users\aksha\Downloads\Supplementary data 1.xlsx')
data.columns = data.columns.str.strip()

if 'SUBJECT_ID' in data.columns:
    data = data.drop('SUBJECT_ID', axis=1)

data = data.fillna(data.mean(numeric_only=True))

#age buckets
bins = [15, 24, 34, 44, 54, 64, 74, 84]
labels = ['AGE: 15-24', 'AGE: 25-34', 'AGE: 35-44', 'AGE: 45-54', 'AGE: 55-64', 'AGE: 65-74', 'AGE: 75-83']
data['Age_Bucket'] = pd.cut(data['Age'], bins=bins, labels=labels, right=True)
data = data.drop('Age', axis=1)

X_num = data.drop(['TYPE', 'Age_Bucket'], axis=1).select_dtypes(include='number')
X_cat = pd.get_dummies(data['Age_Bucket'], drop_first=True)
X = pd.concat([X_num, X_cat], axis=1)

y = data['TYPE']
le = LabelEncoder()
y = le.fit_transform(y)

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#LR tuning
print("Tuning Logistic Regression !")
lr_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'max_iter': [500, 1000, 1500]
}
grid_lr = GridSearchCV(LogisticRegression(), lr_params, cv=5, n_jobs=-1)
grid_lr.fit(X_train_scaled, y_train)
print(f"Best LR params: {grid_lr.best_params_}\n")

#RF tuning
print("Tuning Random Forest !")
rf_params = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)
print(f"Best RF params: {grid_rf.best_params_}\n")

#XGBoost tuning
print("Tuning XGBoost !")
xgb_params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_xgb = GridSearchCV(XGBClassifier(eval_metric='logloss'), xgb_params, cv=5, n_jobs=-1)
grid_xgb.fit(X_train_scaled, y_train)
print(f"Best XGBoost params: {grid_xgb.best_params_}\n")

#model def
models = {
    "Logistic Regression": grid_lr.best_estimator_,
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": grid_rf.best_estimator_,
    "XGBoost": grid_xgb.best_estimator_,
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True)
}

# Stacking ensemble
stacking_estimators = [
    ('lr', grid_lr.best_estimator_),
    ('rf', grid_rf.best_estimator_),
    ('svm', SVC(probability=True)),
    ('nv', GaussianNB()),
    ('xb', grid_xgb.best_estimator_)
]
stack_model = StackingClassifier(estimators=stacking_estimators, final_estimator=RandomForestClassifier())
models["Stacking Classifier"] = stack_model

#traing
print("Model Accuracy Comparison:")
for name, clsfr in models.items():
    clsfr.fit(X_train_scaled, y_train)
    y_pred = clsfr.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.2f}")

#LR coeffs
lr = models['Logistic Regression']
print("\nLogistic Regression Coefficients:")
for coef, feature in zip(lr.coef_[0], X.columns):
    res = "increases risk" if coef > 0 else "decreases risk"
    print(f"{feature}: {coef:.4f} ({res})")

#RF feature imprtance
rf = models['Random Forest']
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nRandom Forest Feature Importances:")
print(importances)

importances.plot(kind='barh', figsize=(12, 8), title='Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


#new data pred
def predict_tumor(new_data):
    """Predict tumor class for new data."""
    model = pickle.load(open('best_model.pkl', 'rb'))
    le_loaded = pickle.load(open('label_encoder.pkl', 'rb'))
    feature_names = pickle.load(open('feature_names.pkl', 'rb'))

    new_df = pd.DataFrame([new_data])

    if 'Age' in new_df.columns:
        new_df['Age_Bucket'] = pd.cut(new_df['Age'], bins=bins, labels=labels, right=True)
        new_df = new_df.drop('Age', axis=1)

    X_new_num = new_df.select_dtypes(include='number')
    if 'Age_Bucket' in new_df.columns:
        X_new_cat = pd.get_dummies(new_df['Age_Bucket'], drop_first=True)
        X_new = pd.concat([X_new_num, X_new_cat], axis=1)
    else:
        X_new = X_new_num

    for col in feature_names:
        if col not in X_new.columns:
            X_new[col] = 0
    X_new = X_new[feature_names]

    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0]

    print("\nTUMOR PREDICTION:")
    print(f"Predicted Class: {le_loaded.inverse_transform([prediction])[0]}")
    print(f"Confidence: {max(probability):.2%}")

    return le_loaded.inverse_transform([prediction])[0]

sample_patient_1 = {
    'AFP': 3.58,
    'AG': 19.36,
    'Age': 47,
    'ALB': 45.4,
    'ALP': 56,
    'ALT': 11,
    'AST': 24,
    'BASO#': 0.01,
    'BASO%': 0.3,
    'BUN': 5.35,
    'Ca': 2.48,
    'CA125': 15.36,
    'CA19-9': 36.48,
    'CA72-4': 6.42,
    'CEA': 1.4,
    'CL': 107.4,
    'CO2': 19.9,
    'CP': 103,
    'CREA': 0,
    'DBIL': 2,
    'EO#': 0.04,
    'EO%': 1,
    'GGT': 16,
    'GLO': 28.5,
    'GLU': 4.67,
    'HCT': 0.273,
    'HE4': 89,
    'HGB': 3.5,
    'IBIL': 5.36,
    'K': 0.65,
    'LYM#': 16.8,
    'LYM%': 33.7,
    'MCH': 103.4,
    'MCV': 0.78,
    'Menopause': 0,
    'Mg': 0.22,
    'MONO#': 5.7,
    'MONO%': 11.7,
    'MPV': 141.3,
    'Na': 76.2,
    'NEU': 0.09,
    'PCT': 13.4,
    'PDW': 1.46,
    'PHOS': 74,
    'PLT': 2.64,
    'RBC': 13.7,
    'RDW': 5.5,
    'TBIL': 73.9,
    'TP': 396.4,
    'UA': 5.5
}

#this sample matches the req type but confidence is low
print("\nPATIENT 1 test (from the dataset)")
print("-" * 60)
predicted_class_1 = predict_tumor(sample_patient_1)
