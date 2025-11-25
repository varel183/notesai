# IF3170 PRAKTIKUM 2 - CHEAT SHEET
**Materi: KNN, DTL, Logistic Regression, SVM**

---

## 1. K-NEAREST NEIGHBORS (KNN)

### Konsep Dasar
- Algoritma lazy learning (tidak ada fase training eksplisit)
- Klasifikasi berdasarkan mayoritas voting dari K tetangga terdekat
- Non-parametric, instance-based learning

### Formula Distance

**Euclidean Distance:**
```
d(p,q) = √(Σ(pi - qi)²)
```

**Manhattan Distance:**
```
d(p,q) = Σ|pi - qi|
```

**Minkowski Distance:**
```
d(p,q) = (Σ|pi - qi|^p)^(1/p)
```
- p=1 → Manhattan
- p=2 → Euclidean

### Weighted KNN
```
weight = 1 / distance
atau
weight = 1 / (distance²)
```

### Hyperparameter
- **K**: jumlah tetangga (biasanya ganjil untuk binary classification)
- **Distance metric**: Euclidean, Manhattan, Minkowski
- **Weights**: uniform atau distance-weighted

### Tips
- ✅ Normalisasi/standardisasi data wajib!
- ✅ K kecil → high variance, low bias
- ✅ K besar → low variance, high bias
- ✅ Optimal K biasanya dicari dengan cross-validation

### Template Kode
```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

# Load dan split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
knn.fit(X_train_scaled, y_train)

# Prediksi
y_pred = knn.predict(X_test_scaled)

# Cross-validation untuk tuning K
k_range = range(1, 31)
cv_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Best K
best_k = k_range[np.argmax(cv_scores)]
```

---

## 2. DECISION TREE LEARNING (DTL)

### Konsep Dasar
- Algoritma divide-and-conquer untuk membuat pohon keputusan
- Setiap node internal = test pada atribut
- Setiap cabang = hasil test
- Setiap leaf = label kelas

### Formula Entropy
```
Entropy(S) = -Σ(pi × log2(pi))
```
- pi = proporsi sampel kelas i
- Entropy = 0 → pure (semua satu kelas)
- Entropy = 1 → maksimal impure (binary, 50-50)

### Information Gain
```
IG(S, A) = Entropy(S) - Σ(|Sv|/|S| × Entropy(Sv))
```
- S = dataset
- A = atribut
- Sv = subset S dimana A = v
- **Pilih atribut dengan IG tertinggi**

### Gain Ratio (C4.5)
```
GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)

SplitInfo(S, A) = -Σ(|Sv|/|S| × log2(|Sv|/|S|))
```
- Mengatasi bias IG terhadap atribut dengan banyak nilai

### Gini Index (CART)
```
Gini(S) = 1 - Σ(pi²)

GiniGain(S, A) = Gini(S) - Σ(|Sv|/|S| × Gini(Sv))
```

### Hyperparameter
- **max_depth**: kedalaman maksimal pohon
- **min_samples_split**: minimum sampel untuk split
- **min_samples_leaf**: minimum sampel di leaf
- **criterion**: 'entropy', 'gini'

### Tips
- ✅ Overfitting prevention: pruning, max_depth, min_samples
- ✅ Categorical features: one-hot encoding atau label encoding
- ✅ Missing values: surrogate splits atau imputation

### Template Kode
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Training
dt = DecisionTreeClassifier(
    criterion='entropy',  # atau 'gini'
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
dt.fit(X_train, y_train)

# Prediksi
y_pred = dt.predict(X_test)

# Visualisasi pohon
plt.figure(figsize=(20,10))
plot_tree(dt, filled=True, feature_names=feature_names, class_names=class_names)
plt.show()

# Feature importance
importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]

# Manual calculation (untuk pemahaman)
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))

def information_gain(X, y, feature_idx, threshold=None):
    parent_entropy = entropy(y)
    
    # Binary split
    if threshold is not None:
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
    else:  # Categorical
        unique_values = np.unique(X[:, feature_idx])
        # Calculate for each split...
    
    # Weighted child entropy
    n = len(y)
    left_entropy = entropy(y[left_mask])
    right_entropy = entropy(y[right_mask])
    
    weighted_entropy = (len(y[left_mask])/n * left_entropy + 
                       len(y[right_mask])/n * right_entropy)
    
    return parent_entropy - weighted_entropy
```

---

## 3. LOGISTIC REGRESSION

### Konsep Dasar
- Model linear untuk klasifikasi (bukan regresi!)
- Output: probabilitas kelas menggunakan fungsi sigmoid
- Binary classification: P(y=1|x)
- Multi-class: One-vs-Rest atau Softmax

### Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))

z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = wᵀx
```

**Properties:**
- Output range: (0, 1)
- σ(0) = 0.5
- σ(∞) = 1
- σ(-∞) = 0

### Hypothesis
```
h(x) = σ(wᵀx) = 1 / (1 + e^(-wᵀx))
```

### Cost Function (Log Loss / Cross-Entropy)
```
J(w) = -(1/m) × Σ[yi×log(h(xi)) + (1-yi)×log(1-h(xi))]
```
- m = jumlah sampel
- yi = label actual (0 atau 1)
- h(xi) = prediksi probabilitas

### Gradient Descent
```
wj := wj - α × ∂J(w)/∂wj

∂J(w)/∂wj = (1/m) × Σ[(h(xi) - yi) × xij]
```
- α = learning rate
- Iterate sampai konvergen

### Regularization

**L2 (Ridge):**
```
J(w) = -(1/m) × Σ[...] + (λ/2m) × Σwj²
```

**L1 (Lasso):**
```
J(w) = -(1/m) × Σ[...] + (λ/m) × Σ|wj|
```

### Decision Boundary
```
Prediksi = 1 jika h(x) ≥ 0.5 (atau threshold lain)
Prediksi = 0 jika h(x) < 0.5
```

### Hyperparameter
- **C**: inverse of regularization strength (sklearn)
  - C kecil → regularisasi kuat
  - C besar → regularisasi lemah
- **penalty**: 'l1', 'l2', 'elasticnet'
- **solver**: 'lbfgs', 'liblinear', 'saga'
- **max_iter**: iterasi maksimal

### Tips
- ✅ Feature scaling penting (gradient descent lebih cepat)
- ✅ Handle imbalanced data: class_weight='balanced'
- ✅ Multi-class: multi_class='ovr' atau 'multinomial'

### Template Kode
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Normalisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training
logreg = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
logreg.fit(X_train_scaled, y_train)

# Prediksi
y_pred = logreg.predict(X_test_scaled)
y_pred_proba = logreg.predict_proba(X_test_scaled)  # Probabilitas

# Coefficients
print("Intercept:", logreg.intercept_)
print("Coefficients:", logreg.coef_)

# Manual implementation (untuk pemahaman)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_gd(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    
    for i in range(iterations):
        # Forward pass
        z = np.dot(X, w) + b
        predictions = sigmoid(z)
        
        # Compute gradients
        dw = (1/m) * np.dot(X.T, (predictions - y))
        db = (1/m) * np.sum(predictions - y)
        
        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # Compute cost (optional, for monitoring)
        if i % 100 == 0:
            cost = -(1/m) * np.sum(y*np.log(predictions + 1e-9) + 
                                   (1-y)*np.log(1-predictions + 1e-9))
            print(f"Iteration {i}, Cost: {cost}")
    
    return w, b

# Tuning C dengan cross-validation
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)
print("Best C:", grid.best_params_)
```

---

## 4. SUPPORT VECTOR MACHINE (SVM)

### Konsep Dasar
- Mencari hyperplane optimal yang memisahkan kelas dengan margin maksimal
- Support vectors = data points terdekat dengan hyperplane
- Margin = jarak dari hyperplane ke support vector terdekat
- **Goal: Maximize margin**

### Hyperplane
```
wᵀx + b = 0
```
- w = weight vector (normal ke hyperplane)
- b = bias
- Dimensi hyperplane = n-1 (untuk data n-dimensi)

### Decision Function
```
f(x) = sign(wᵀx + b)
```
- f(x) = +1 jika wᵀx + b ≥ 0
- f(x) = -1 jika wᵀx + b < 0

### Margin
```
Margin = 2 / ||w||

Maximize Margin ⟺ Minimize ||w||²/2
```

### Primal Optimization Problem
```
Minimize: (1/2)||w||²

Subject to: yi(wᵀxi + b) ≥ 1, untuk semua i
```
- yi ∈ {-1, +1}

### Soft Margin (dengan slack variables)
```
Minimize: (1/2)||w||² + C×Σξi

Subject to: 
- yi(wᵀxi + b) ≥ 1 - ξi
- ξi ≥ 0
```
- ξi = slack variable (toleransi misklasifikasi)
- C = regularization parameter
  - C besar → hard margin (sedikit toleransi)
  - C kecil → soft margin (banyak toleransi)

### Kernel Functions

**Linear Kernel:**
```
K(x, x') = xᵀx'
```

**Polynomial Kernel:**
```
K(x, x') = (γxᵀx' + r)^d
```
- d = degree
- γ = gamma
- r = coef0

**RBF (Radial Basis Function) / Gaussian Kernel:**
```
K(x, x') = exp(-γ||x - x'||²)
```
- γ = gamma parameter
- γ besar → overfitting (kompleks)
- γ kecil → underfitting (sederhana)

**Sigmoid Kernel:**
```
K(x, x') = tanh(γxᵀx' + r)
```

### Hyperparameter

**C (Regularization):**
- Mengontrol trade-off antara margin dan error
- C ↑ → margin lebih sempit, lebih sedikit misklasifikasi
- C ↓ → margin lebih lebar, toleran terhadap misklasifikasi

**Gamma (untuk RBF/Poly):**
- Mengontrol jangkauan pengaruh single training example
- γ ↑ → jangkauan pendek (kompleks, overfitting risk)
- γ ↓ → jangkauan jauh (sederhana, underfitting risk)

**Kernel:**
- Linear: data linearly separable
- Polynomial: interaksi non-linear sedang
- RBF: non-linear kompleks (paling umum)

### Tips
- ✅ Feature scaling WAJIB (SVM sensitive terhadap scale)
- ✅ Untuk data besar: LinearSVC lebih cepat
- ✅ RBF kernel: default choice untuk non-linear
- ✅ Tuning C dan gamma sangat penting

### Template Kode
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Normalisasi (WAJIB!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training - Linear SVM
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_scaled, y_train)

# Training - RBF SVM
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)

# Training - Polynomial SVM
svm_poly = SVC(kernel='poly', degree=3, C=1.0, gamma='scale', random_state=42)
svm_poly.fit(X_train_scaled, y_train)

# Prediksi
y_pred = svm_rbf.predict(X_test_scaled)

# Support vectors
print("Number of support vectors:", len(svm_rbf.support_))
print("Support vector indices:", svm_rbf.support_)
print("Support vectors per class:", svm_rbf.n_support_)

# Hyperparameter tuning dengan GridSearch
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_scaled, y_train)

print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)

# Decision boundary visualization (2D only)
def plot_decision_boundary(model, X, y):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()

# Jika data 2D
if X_train_scaled.shape[1] == 2:
    plot_decision_boundary(svm_rbf, X_train_scaled, y_train)
```

---

## 5. EVALUATION METRICS

### Confusion Matrix
```
                Predicted
              Positive  Negative
Actual Pos      TP        FN
       Neg      FP        TN
```

### Metrics Formula

**Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision:**
```
Precision = TP / (TP + FP)
```
- Dari yang diprediksi positif, berapa yang benar?

**Recall (Sensitivity, TPR):**
```
Recall = TP / (TP + FN)
```
- Dari yang actual positif, berapa yang terdeteksi?

**F1-Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Harmonic mean dari precision dan recall

**Specificity (TNR):**
```
Specificity = TN / (TN + FP)
```

**False Positive Rate:**
```
FPR = FP / (FP + TN) = 1 - Specificity
```

### Template Kode
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

# ROC Curve (binary classification)
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Cross-validation scores
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

---

## 6. DATA PREPROCESSING

### Missing Values
```python
from sklearn.impute import SimpleImputer

# Mean imputation
imputer = SimpleImputer(strategy='mean')  # atau 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X)
```

### Feature Scaling

**Standardization (Z-score):**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Formula: z = (x - μ) / σ
```

**Normalization (Min-Max):**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_normalized = scaler.fit_transform(X_train)

# Formula: x' = (x - min) / (max - min)
```

### Encoding Categorical Variables

**Label Encoding:**
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)
```

**One-Hot Encoding:**
```python
import pandas as pd

X_encoded = pd.get_dummies(X, columns=['categorical_column'])

# Atau sklearn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X[['categorical_column']])
```

### Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 80-20 split
    random_state=42,    # reproducibility
    stratify=y          # maintain class distribution
)
```

### Handling Imbalanced Data
```python
# Class weights
model = LogisticRegression(class_weight='balanced')

# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

---

## 7. PERBANDINGAN ALGORITMA

### Kapan Menggunakan Apa?

| Algoritma | Kelebihan | Kekurangan | Best For |
|-----------|-----------|------------|----------|
| **KNN** | - Sederhana<br>- No training<br>- Non-parametric | - Slow prediction<br>- Sensitive to scale<br>- Curse of dimensionality | - Small dataset<br>- Non-linear boundary<br>- Multi-class |
| **Decision Tree** | - Interpretable<br>- No scaling needed<br>- Handle non-linear | - Overfitting prone<br>- Unstable<br>- Bias pada imbalanced | - Interpretability penting<br>- Mixed data types<br>- Feature importance |
| **Logistic Reg** | - Fast<br>- Probabilistic output<br>- Regularization | - Linear boundary<br>- Feature engineering needed | - Binary classification<br>- Linearly separable<br>- Need probability |
| **SVM** | - Effective high-dim<br>- Memory efficient<br>- Versatile kernels | - Slow on large data<br>- Sensitive to params<br>- Need scaling | - High-dimensional<br>- Non-linear (RBF)<br>- Clear margin |

### Complexity

| Algoritma | Training Time | Prediction Time | Space |
|-----------|---------------|-----------------|-------|
| KNN | O(1) | O(nd) | O(nd) |
| Decision Tree | O(n×d×log n) | O(log n) | O(nodes) |
| Logistic Reg | O(n×d×iter) | O(d) | O(d) |
| SVM | O(n²×d) to O(n³×d) | O(sv×d) | O(sv×d) |

n = samples, d = features, sv = support vectors

---

## 8. COMPLETE WORKFLOW TEMPLATE

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ========== 1. LOAD DATA ==========
# df = pd.read_csv('data.csv')
# X = df.drop('target', axis=1)
# y = df['target']

# ========== 2. PREPROCESSING ==========
# Handle missing values
# X = X.fillna(X.mean())

# Encode categorical
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 3. MODELS ==========
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

# ========== 4. TRAINING & EVALUATION ==========
results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    print(f"{'='*50}")
    
    # Train
    if name in ['KNN', 'Logistic Regression', 'SVM']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:  # Decision Tree (no scaling)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    }
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{name}_confusion_matrix.png')
    plt.close()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# ========== 5. COMPARISON ==========
results_df = pd.DataFrame(results).T
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(results_df)

# Plot comparison
results_df.plot(kind='bar', figsize=(12, 6))
plt.title('Model Comparison')
plt.xlabel('Model')
plt.ylabel('Score')
plt.legend(loc='lower right')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# ========== 6. HYPERPARAMETER TUNING (Optional) ==========
# Example: Tuning KNN
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_knn = GridSearchCV(
    KNeighborsClassifier(), 
    param_grid_knn, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1
)
grid_knn.fit(X_train_scaled, y_train)

print("\nBest KNN Parameters:", grid_knn.best_params_)
print("Best KNN Score:", grid_knn.best_score_)

# Example: Tuning SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'kernel': ['rbf', 'poly']
}

grid_svm = GridSearchCV(
    SVC(), 
    param_grid_svm, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1
)
grid_svm.fit(X_train_scaled, y_train)

print("\nBest SVM Parameters:", grid_svm.best_params_)
print("Best SVM Score:", grid_svm.best_score_)

# ========== 7. FINAL MODEL ==========
# Retrain dengan best parameters
best_model = grid_svm.best_estimator_
y_pred_final = best_model.predict(X_test_scaled)

print("\n" + "="*60)
print("FINAL MODEL PERFORMANCE")
print("="*60)
print(classification_report(y_test, y_pred_final))
```

---

## 9. TIPS PENGERJAAN PRAKTIKUM

### Timeline Strategy
1. **09.00 - Deadline 1 (Kode Program)**
   - Implementasi 4 algoritma
   - Basic evaluation metrics
   - Pastikan semua running

2. **09.00 - 22.00 (Analisis)**
   - Hyperparameter tuning
   - Detailed comparison
   - Visualisasi
   - Interpretasi hasil

### Checklist Kode Program
- [ ] Data loading & exploration
- [ ] Preprocessing (scaling, encoding, split)
- [ ] KNN implementation
- [ ] Decision Tree implementation
- [ ] Logistic Regression implementation
- [ ] SVM implementation
- [ ] Evaluation metrics (accuracy, precision, recall, F1)
- [ ] Confusion matrix
- [ ] Model comparison table

### Checklist Analisis
- [ ] Deskripsi dataset
- [ ] Preprocessing steps & rationale
- [ ] Hyperparameter tuning process
- [ ] Performance comparison (table + grafik)
- [ ] Confusion matrix analysis
- [ ] Feature importance (jika ada)
- [ ] Best model selection & justification
- [ ] Kelebihan/kekurangan tiap algoritma pada dataset
- [ ] Kesimpulan

### Common Pitfalls to Avoid
❌ Lupa scaling untuk KNN, LogReg, SVM
❌ Tidak stratify saat split (untuk imbalanced data)
❌ Overfit karena tidak validation
❌ Tidak handle missing values
❌ Comparison tidak fair (beda preprocessing)

### Quick Debug Commands
```python
# Check data shape
print(X.shape, y.shape)

# Check missing values
print(X.isnull().sum())

# Check class distribution
print(y.value_counts())

# Check if scaled properly
print(X_scaled.mean(axis=0))  # Should be ~0
print(X_scaled.std(axis=0))   # Should be ~1

# Check predictions shape
print(y_pred.shape, y_test.shape)
```

---

## 10. QUICK REFERENCE - SKLEARN PARAMETERS

### KNeighborsClassifier
```python
KNeighborsClassifier(
    n_neighbors=5,        # K value
    weights='uniform',    # 'uniform' or 'distance'
    metric='euclidean',   # 'euclidean', 'manhattan', 'minkowski'
    p=2                   # Power for Minkowski
)
```

### DecisionTreeClassifier
```python
DecisionTreeClassifier(
    criterion='gini',     # 'gini' or 'entropy'
    max_depth=None,       # Max tree depth
    min_samples_split=2,  # Min samples to split
    min_samples_leaf=1,   # Min samples in leaf
    random_state=42
)
```

### LogisticRegression
```python
LogisticRegression(
    penalty='l2',         # 'l1', 'l2', 'elasticnet', 'none'
    C=1.0,               # Inverse regularization strength
    solver='lbfgs',      # 'lbfgs', 'liblinear', 'saga'
    max_iter=100,
    random_state=42
)
```

### SVC
```python
SVC(
    C=1.0,               # Regularization parameter
    kernel='rbf',        # 'linear', 'poly', 'rbf', 'sigmoid'
    degree=3,            # Degree for poly kernel
    gamma='scale',       # 'scale', 'auto', or float
    random_state=42
)
```

---

## GOOD LUCK! 🚀

**Remember:**
- ✅ Normalisasi untuk KNN, LogReg, SVM
- ✅ Cross-validation untuk tuning
- ✅ Stratify split untuk imbalanced data
- ✅ Compare apples to apples (sama preprocessing)
- ✅ Interpretasi hasil, bukan cuma angka
- ✅ Dokumentasi setiap langkah

**Jangan lupa isi pembagian kelompok sebelum besok pukul 09.00!**
