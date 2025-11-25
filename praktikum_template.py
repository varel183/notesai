"""
IF3170 PRAKTIKUM 2 - MACHINE LEARNING ALGORITHMS
Materi: KNN, Decision Tree, Logistic Regression, SVM

Author: [Nama Kelompok]
Tanggal: 26 November 2025

DESKRIPSI:
Template ini berisi implementasi lengkap 4 algoritma machine learning:
1. K-Nearest Neighbors (KNN)
2. Decision Tree Learning (DTL)
3. Logistic Regression
4. Support Vector Machine (SVM)

Lengkap dengan preprocessing, evaluation, visualization, dan comparison.
"""

# ============================================================================
# IMPORT LIBRARIES
# ============================================================================
# Library untuk manipulasi data dan numerik
import numpy as np          # Operasi array dan matematika
import pandas as pd         # Manipulasi dataframe
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns       # Visualisasi statistik
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings agar output lebih bersih

# Library Machine Learning dari Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# - train_test_split: Split data train-test
# - cross_val_score: Cross-validation untuk evaluasi model
# - GridSearchCV: Hyperparameter tuning

from sklearn.preprocessing import StandardScaler, LabelEncoder
# - StandardScaler: Normalisasi fitur (mean=0, std=1)
# - LabelEncoder: Encode categorical target menjadi numeric

from sklearn.impute import SimpleImputer
# - SimpleImputer: Handle missing values

# Import 4 Algoritma yang akan digunakan
from sklearn.neighbors import KNeighborsClassifier      # KNN
from sklearn.tree import DecisionTreeClassifier, plot_tree  # Decision Tree
from sklearn.linear_model import LogisticRegression     # Logistic Regression
from sklearn.svm import SVC                             # SVM

# Library untuk Evaluasi Model
from sklearn.metrics import (
    accuracy_score,         # Akurasi = (TP+TN)/(TP+TN+FP+FN)
    precision_score,        # Precision = TP/(TP+FP)
    recall_score,           # Recall = TP/(TP+FN)
    f1_score,              # F1 = 2*(Precision*Recall)/(Precision+Recall)
    confusion_matrix,       # Matrix TP, TN, FP, FN
    classification_report,  # Report lengkap per class
    roc_curve,             # ROC curve untuk binary classification
    auc                    # Area Under Curve
)

# Set random seed untuk reproducibility
# Dengan seed yang sama, hasil random akan selalu sama
np.random.seed(42)

# Set style plotting agar grafik lebih menarik
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ All libraries imported successfully!")


# ============================================================================
# 1. LOAD & EXPLORE DATA
# ============================================================================
print("\n" + "="*80)
print("1. DATA LOADING & EXPLORATION")
print("="*80)

# ----------------------------------------------------------------------------
# TODO: GANTI BAGIAN INI DENGAN DATA KAMU
# ----------------------------------------------------------------------------
# Uncomment dan sesuaikan dengan format data kamu:
# 
# CARA 1: Load dari CSV
# df = pd.read_csv('your_data.csv')
# X = df.drop('target_column', axis=1)  # Fitur (semua kolom kecuali target)
# y = df['target_column']                # Target/label
#
# CARA 2: Load dari Excel
# df = pd.read_excel('your_data.xlsx')
# X = df.drop('target_column', axis=1)
# y = df['target_column']
#
# CARA 3: Jika fitur dan target sudah terpisah
# X = pd.read_csv('features.csv')
# y = pd.read_csv('labels.csv')
# ----------------------------------------------------------------------------

# Untuk demonstrasi, kita pakai dataset Iris (HAPUS INI NANTI!)
from sklearn.datasets import load_iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Tampilkan informasi dataset
print(f"\nDataset shape: {X.shape}")  # (jumlah_sampel, jumlah_fitur)
print(f"Target shape: {y.shape}")     # (jumlah_sampel,)
print(f"\nFirst 5 rows:")
print(X.head())  # Lihat 5 baris pertama untuk memahami struktur data

# Cek apakah ada missing values (data yang hilang)
print(f"\nMissing values:")
print(X.isnull().sum())  # Jika ada yang > 0, perlu dihandle di preprocessing

# Lihat distribusi target/label
# Penting untuk tau apakah data balanced atau imbalanced
print(f"\nTarget distribution:")
print(y.value_counts())  # Jumlah sampel per class

# Statistik deskriptif (mean, std, min, max, dll)
# Berguna untuk deteksi outlier dan understand range data
print(f"\nBasic statistics:")
print(X.describe())

# Visualisasi distribusi target
# Grafik ini menunjukkan apakah ada class imbalance
plt.figure(figsize=(8, 5))
y.value_counts().plot(kind='bar')
plt.title('Target Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: target_distribution.png")


# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("2. DATA PREPROCESSING")
print("="*80)

# ----------------------------------------------------------------------------
# STEP 1: Handle Missing Values
# ----------------------------------------------------------------------------
# Missing values bisa menyebabkan error saat training
# Ada beberapa strategi: mean, median, most_frequent, atau drop
if X.isnull().sum().sum() > 0:
    print("\nHandling missing values...")
    imputer = SimpleImputer(strategy='mean')  # Isi dengan rata-rata
    # Strategi lain:
    # - strategy='median': isi dengan median (untuk data skewed)
    # - strategy='most_frequent': isi dengan modus (untuk categorical)
    # - strategy='constant', fill_value=0: isi dengan nilai tertentu
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    print("✓ Missing values handled")
else:
    print("\n✓ No missing values found")

# ----------------------------------------------------------------------------
# STEP 2: Encode Categorical Target
# ----------------------------------------------------------------------------
# Jika target berupa string (misal: 'setosa', 'versicolor', 'virginica')
# Perlu diubah jadi angka (0, 1, 2)
if y.dtype == 'object':
    print("\nEncoding target variable...")
    le = LabelEncoder()
    y = le.fit_transform(y)
    print("✓ Target encoded")
    print(f"  Classes: {le.classes_}")  # Lihat mapping class ke angka
else:
    print("\n✓ Target already numeric")

# ----------------------------------------------------------------------------
# STEP 3: Train-Test Split
# ----------------------------------------------------------------------------
# Split data menjadi training set (80%) dan testing set (20%)
# Training set: untuk melatih model
# Testing set: untuk evaluasi performa model (data yang belum pernah dilihat)
print("\nSplitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% untuk testing
    random_state=42,    # Seed untuk reproducibility
    stratify=y          # Maintain proporsi class di train dan test
)
# stratify=y penting untuk imbalanced data
# Contoh: jika class A=80%, B=20%, maka di train dan test juga 80%-20%

print(f"Train set: {X_train.shape} samples")
print(f"Test set: {X_test.shape} samples")

# ----------------------------------------------------------------------------
# STEP 4: Feature Scaling
# ----------------------------------------------------------------------------
# Feature scaling WAJIB untuk KNN, Logistic Regression, dan SVM
# Karena algoritma ini sensitive terhadap skala fitur
# 
# Contoh kenapa penting:
# - Fitur 1: Income (range 1000-100000)
# - Fitur 2: Age (range 18-65)
# Tanpa scaling, Income akan dominan karena nilai lebih besar
#
# StandardScaler: z = (x - mean) / std
# Hasil: mean=0, std=1 untuk setiap fitur

print("\nScaling features...")
scaler = StandardScaler()

# PENTING: fit_transform di training, transform di testing
# fit_transform: hitung mean & std dari train, lalu transform
X_train_scaled = scaler.fit_transform(X_train)

# transform: gunakan mean & std dari train untuk transform test
# JANGAN fit di test set! (data leakage)
X_test_scaled = scaler.transform(X_test)

# Verify scaling berhasil
mean_check = X_train_scaled.mean(axis=0)[:3]  # Ambil 3 fitur pertama
std_check = X_train_scaled.std(axis=0)[:3]
print(f"Mean of scaled features (should be ~0): {mean_check}")
print(f"Std of scaled features (should be ~1): {std_check}")
print("✓ Preprocessing complete!")

# NOTE: Decision Tree TIDAK perlu scaling
# Karena DT tidak menggunakan distance/magnitude, hanya split points


# ============================================================================
# 3. MODEL TRAINING & EVALUATION
# ============================================================================
print("\n" + "="*80)
print("3. MODEL TRAINING & EVALUATION")
print("="*80)

# Dictionary untuk menyimpan hasil evaluasi semua model
results = {}
# Dictionary untuk menyimpan trained model (jika perlu pakai lagi nanti)
trained_models = {}


# ----------------------------------------------------------------------------
# 3.1 K-NEAREST NEIGHBORS (KNN)
# ----------------------------------------------------------------------------
print("\n" + "-"*80)
print("3.1 K-NEAREST NEIGHBORS (KNN)")
print("-"*80)

# KONSEP KNN:
# - Lazy learning: tidak ada fase training eksplisit
# - Klasifikasi berdasarkan majority voting dari K tetangga terdekat
# - Distance metric: Euclidean (default), Manhattan, Minkowski
# - Perlu feature scaling karena sensitive terhadap skala

# Inisialisasi model KNN
knn = KNeighborsClassifier(
    n_neighbors=5,       # K = jumlah tetangga yang dilihat
                        # K kecil → complex (overfitting risk)
                        # K besar → simple (underfitting risk)
                        # Rule of thumb: K = sqrt(n_samples) atau coba 3,5,7,9
    
    weights='uniform',   # Semua tetangga punya voting power sama
                        # 'distance': tetangga lebih dekat = voting lebih kuat
    
    metric='euclidean'   # Distance metric
                        # 'euclidean': √(Σ(x-y)²)
                        # 'manhattan': Σ|x-y|
                        # 'minkowski': generalisasi keduanya
)

# Training KNN (sebenarnya hanya menyimpan data training)
knn.fit(X_train_scaled, y_train)  # WAJIB pakai data yang sudah di-scale!

# Prediksi pada test set
y_pred_knn = knn.predict(X_test_scaled)  # WAJIB scale dulu!

# Evaluasi performa model
# Accuracy: berapa persen prediksi yang benar
acc_knn = accuracy_score(y_test, y_pred_knn)

# Precision: dari yang diprediksi positif, berapa yang bener positif?
# Important untuk minimize False Positive
prec_knn = precision_score(y_test, y_pred_knn, average='weighted', zero_division=0)

# Recall: dari yang actual positif, berapa yang berhasil terdeteksi?
# Important untuk minimize False Negative
rec_knn = recall_score(y_test, y_pred_knn, average='weighted')

# F1-Score: harmonic mean dari precision & recall
# Bagus untuk imbalanced dataset
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')

# Simpan hasil
results['KNN'] = {
    'Accuracy': acc_knn,
    'Precision': prec_knn,
    'Recall': rec_knn,
    'F1-Score': f1_knn
}
trained_models['KNN'] = knn

# Print hasil
print(f"Accuracy:  {acc_knn:.4f}")   # Semakin tinggi semakin baik
print(f"Precision: {prec_knn:.4f}")  # Semakin tinggi semakin baik
print(f"Recall:    {rec_knn:.4f}")   # Semakin tinggi semakin baik
print(f"F1-Score:  {f1_knn:.4f}")    # Semakin tinggi semakin baik

# Confusion Matrix
# Matrix yang menunjukkan:
# - TP (True Positive): diprediksi positif, actual positif ✓
# - TN (True Negative): diprediksi negatif, actual negatif ✓
# - FP (False Positive): diprediksi positif, actual negatif ✗
# - FN (False Negative): diprediksi negatif, actual positif ✗
cm_knn = confusion_matrix(y_test, y_pred_knn)

# Visualisasi confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('KNN - Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('knn_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: knn_confusion_matrix.png")

# Classification Report (detail per class)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

# ----------------------------------------------------------------------------
# BONUS: Find Optimal K
# ----------------------------------------------------------------------------
# Kita coba berbagai nilai K untuk cari yang terbaik
# Gunakan cross-validation untuk evaluasi yang lebih robust
print("\nFinding optimal K...")
k_range = range(1, 21)  # Coba K dari 1 sampai 20
cv_scores = []

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    # Cross-validation: split train jadi 5 fold, rata-rata accuracy
    scores = cross_val_score(knn_temp, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# K dengan CV score tertinggi
optimal_k = k_range[np.argmax(cv_scores)]
print(f"Optimal K: {optimal_k} (CV Score: {max(cv_scores):.4f})")

# Plot K vs Accuracy untuk visualisasi
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores, marker='o', linestyle='-', linewidth=2)
plt.xlabel('K Value', fontsize=12)
plt.ylabel('Cross-Validation Accuracy', fontsize=12)
plt.title('KNN: K Value vs Accuracy', fontsize=14, fontweight='bold')
plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('knn_k_optimization.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: knn_k_optimization.png")


# ----------------------------------------------------------------------------
# 3.2 DECISION TREE
# ----------------------------------------------------------------------------
print("\n" + "-"*80)
print("3.2 DECISION TREE")
print("-"*80)

# KONSEP DECISION TREE:
# - Membuat pohon keputusan dengan split berdasarkan fitur
# - Setiap node internal = test pada atribut
# - Setiap leaf = label kelas
# - Split criterion: Information Gain (entropy) atau Gini Index
# - TIDAK perlu feature scaling!

# Inisialisasi Decision Tree
dt = DecisionTreeClassifier(
    criterion='entropy',      # Splitting criterion
                             # 'entropy': Information Gain (ID3/C4.5)
                             #   Entropy = -Σ(pi × log2(pi))
                             #   IG = Entropy(parent) - weighted_avg(Entropy(children))
                             # 'gini': Gini Impurity (CART)
                             #   Gini = 1 - Σ(pi²)
    
    max_depth=5,             # Kedalaman maksimal pohon
                             # Kecil → simple, underfitting risk
                             # Besar → complex, overfitting risk
                             # None → grow sampai semua leaf pure (overfitting!)
    
    min_samples_split=10,    # Min sampel untuk split node
                             # Besar → pohon lebih sederhana (pruning)
    
    min_samples_leaf=5,      # Min sampel di leaf node
                             # Besar → pohon lebih sederhana (pruning)
    
    random_state=42          # Untuk reproducibility
)

# Training Decision Tree
# PERHATIKAN: Pakai X_train (BUKAN X_train_scaled)
# DT tidak butuh scaling karena hanya melihat split points, bukan distance
dt.fit(X_train, y_train)

# Prediksi
y_pred_dt = dt.predict(X_test)  # Juga tanpa scaling

# Evaluasi
acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt, average='weighted', zero_division=0)
rec_dt = recall_score(y_test, y_pred_dt, average='weighted')
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')

# Simpan hasil
results['Decision Tree'] = {
    'Accuracy': acc_dt,
    'Precision': prec_dt,
    'Recall': rec_dt,
    'F1-Score': f1_dt
}
trained_models['Decision Tree'] = dt

# Print hasil
print(f"Accuracy:  {acc_dt:.4f}")
print(f"Precision: {prec_dt:.4f}")
print(f"Recall:    {rec_dt:.4f}")
print(f"F1-Score:  {f1_dt:.4f}")

# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Decision Tree - Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('dt_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: dt_confusion_matrix.png")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

# ----------------------------------------------------------------------------
# Feature Importance
# ----------------------------------------------------------------------------
# Decision Tree memberikan importance score untuk setiap fitur
# Score menunjukkan seberapa berguna fitur untuk split
# Tinggi = fitur penting untuk klasifikasi

feature_importance = dt.feature_importances_
feature_names = X.columns if isinstance(X, pd.DataFrame) else [f'Feature {i}' for i in range(X.shape[1])]

# Buat dataframe untuk sorting
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (descending):")
print(importance_df)

# Visualisasi feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='green', alpha=0.7)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Decision Tree - Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()  # Fitur terpenting di atas
plt.tight_layout()
plt.savefig('dt_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: dt_feature_importance.png")

# ----------------------------------------------------------------------------
# Visualize Tree Structure
# ----------------------------------------------------------------------------
# Visualisasi struktur pohon (hanya jika tidak terlalu dalam)
if dt.get_depth() <= 5:
    print(f"\nTree depth: {dt.get_depth()}")
    print("Visualizing tree structure...")
    
    plt.figure(figsize=(20, 10))
    plot_tree(
        dt, 
        filled=True,                    # Warna berdasarkan majority class
        feature_names=feature_names,    # Nama fitur
        class_names=[str(c) for c in np.unique(y)],  # Nama class
        rounded=True,                   # Rounded box
        fontsize=10
    )
    plt.title('Decision Tree Structure', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dt_tree_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: dt_tree_visualization.png")
else:
    print(f"\nTree too deep ({dt.get_depth()} levels) - skipping visualization")
    print("Consider reducing max_depth for better interpretability")


# ----------------------------------------------------------------------------
# 3.3 LOGISTIC REGRESSION
# ----------------------------------------------------------------------------
print("\n" + "-"*80)
print("3.3 LOGISTIC REGRESSION")
print("-"*80)

# KONSEP LOGISTIC REGRESSION:
# - Model LINEAR untuk klasifikasi (bukan regresi!)
# - Menggunakan fungsi sigmoid untuk output probabilitas
# - Sigmoid: σ(z) = 1 / (1 + e^(-z)), dimana z = w·x + b
# - Output: P(y=1|x) ∈ [0, 1]
# - Decision boundary: linear (bisa non-linear dengan feature engineering)
# - Optimization: Gradient Descent
# - Loss function: Cross-Entropy/Log Loss
# - PERLU feature scaling!

# Inisialisasi Logistic Regression
logreg = LogisticRegression(
    penalty='l2',            # Regularization type
                            # 'l2': Ridge (Σw²) - default, good untuk multicollinearity
                            # 'l1': Lasso (Σ|w|) - feature selection
                            # 'none': no regularization
    
    C=1.0,                  # Inverse of regularization strength
                            # C besar → weak regularization (bisa overfit)
                            # C kecil → strong regularization (bisa underfit)
                            # Default C=1.0 biasanya sudah cukup
    
    solver='lbfgs',         # Optimization algorithm
                            # 'lbfgs': good untuk small dataset
                            # 'liblinear': good untuk small dataset, support l1
                            # 'saga': good untuk large dataset, support l1
                            # 'newton-cg', 'sag': alternatif lain
    
    max_iter=1000,          # Max iterasi untuk convergence
                            # Kalau warning "max_iter reached", naikkan ini
    
    random_state=42         # Reproducibility
)

# Training Logistic Regression
# WAJIB pakai data scaled!
logreg.fit(X_train_scaled, y_train)

# Prediksi
y_pred_logreg = logreg.predict(X_test_scaled)  # Prediksi label (0 atau 1)
# logreg.predict_proba(X_test_scaled)  # Jika butuh probabilitas

# Evaluasi
acc_logreg = accuracy_score(y_test, y_pred_logreg)
prec_logreg = precision_score(y_test, y_pred_logreg, average='weighted', zero_division=0)
rec_logreg = recall_score(y_test, y_pred_logreg, average='weighted')
f1_logreg = f1_score(y_test, y_pred_logreg, average='weighted')

# Simpan hasil
results['Logistic Regression'] = {
    'Accuracy': acc_logreg,
    'Precision': prec_logreg,
    'Recall': rec_logreg,
    'F1-Score': f1_logreg
}
trained_models['Logistic Regression'] = logreg

# Print hasil
print(f"Accuracy:  {acc_logreg:.4f}")
print(f"Precision: {prec_logreg:.4f}")
print(f"Recall:    {rec_logreg:.4f}")
print(f"F1-Score:  {f1_logreg:.4f}")

# Confusion Matrix
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Oranges', cbar=False)
plt.title('Logistic Regression - Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('logreg_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: logreg_confusion_matrix.png")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_logreg))

# ----------------------------------------------------------------------------
# Model Coefficients (Weights)
# ----------------------------------------------------------------------------
# Coefficients menunjukkan pengaruh setiap fitur terhadap prediksi
# - Positif: fitur naik → probabilitas class 1 naik
# - Negatif: fitur naik → probabilitas class 1 turun
# - Magnitude besar: fitur penting
print("\nModel Parameters:")
print(f"Intercept (bias): {logreg.intercept_}")

# Untuk binary classification, coef_ shape = (1, n_features)
# Untuk multi-class, coef_ shape = (n_classes, n_features)
if logreg.coef_.shape[0] == 1:
    # Binary classification
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': logreg.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)
    print("\nFeature Coefficients (sorted by absolute value):")
    print(coef_df)
    
    # Visualisasi coefficients
    plt.figure(figsize=(10, 6))
    colors = ['red' if c < 0 else 'green' for c in coef_df['Coefficient']]
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Logistic Regression - Feature Coefficients', fontsize=14, fontweight='bold')
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('logreg_coefficients.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: logreg_coefficients.png")
else:
    # Multi-class classification
    print(f"\nCoefficients shape: {logreg.coef_.shape}")
    print("(n_classes, n_features) - one set of coefficients per class")
    for i, class_coef in enumerate(logreg.coef_):
        print(f"\nClass {i} coefficients:")
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': class_coef
        }).sort_values('Coefficient', key=abs, ascending=False)
        print(coef_df.head())  # Top 5 most important features per class


# ----------------------------------------------------------------------------
# 3.4 SUPPORT VECTOR MACHINE (SVM)
# ----------------------------------------------------------------------------
print("\n" + "-"*80)
print("3.4 SUPPORT VECTOR MACHINE (SVM)")
print("-"*80)

# KONSEP SVM:
# - Mencari hyperplane optimal yang memisahkan kelas dengan margin maksimal
# - Hyperplane: w·x + b = 0
# - Margin: jarak dari hyperplane ke support vector terdekat
# - Support vectors: data points yang menentukan hyperplane
# - Objective: Maximize margin ⟺ Minimize ||w||²
# - Kernel trick: mapping ke higher dimension untuk non-linear boundary
# - SANGAT sensitive terhadap feature scale, WAJIB scaling!

# Inisialisasi SVM
svm = SVC(
    kernel='rbf',           # Kernel function
                           # 'linear': K(x,x') = x·x'
                           #   → Linear decision boundary
                           #   → Fast, interpretable
                           #   → Good untuk linearly separable data
                           #
                           # 'rbf' (Radial Basis Function): K(x,x') = exp(-γ||x-x'||²)
                           #   → Non-linear boundary (Gaussian)
                           #   → Default choice, most versatile
                           #   → Good untuk complex, non-linear data
                           #
                           # 'poly': K(x,x') = (γx·x' + r)^d
                           #   → Polynomial boundary
                           #   → degree parameter controls complexity
                           #
                           # 'sigmoid': K(x,x') = tanh(γx·x' + r)
                           #   → Neural network-like
    
    C=1.0,                 # Regularization parameter (soft margin)
                           # Trade-off antara margin width dan misclassification
                           # C besar → hard margin, less tolerance (overfit risk)
                           # C kecil → soft margin, more tolerance (underfit risk)
                           # Typical values: 0.1, 1, 10, 100
    
    gamma='scale',         # Kernel coefficient (untuk rbf, poly, sigmoid)
                           # Mengontrol "jangkauan" pengaruh single training example
                           # gamma besar → jangkauan pendek, complex (overfit risk)
                           # gamma kecil → jangkauan jauh, smooth (underfit risk)
                           # 'scale': 1/(n_features × variance(X)) - default
                           # 'auto': 1/n_features
    
    random_state=42        # Reproducibility
)

# Training SVM
# WAJIB pakai data scaled! SVM SANGAT sensitive terhadap scale
svm.fit(X_train_scaled, y_train)

# Prediksi
y_pred_svm = svm.predict(X_test_scaled)

# Evaluasi
acc_svm = accuracy_score(y_test, y_pred_svm)
prec_svm = precision_score(y_test, y_pred_svm, average='weighted', zero_division=0)
rec_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

# Simpan hasil
results['SVM'] = {
    'Accuracy': acc_svm,
    'Precision': prec_svm,
    'Recall': rec_svm,
    'F1-Score': f1_svm
}
trained_models['SVM'] = svm

# Print hasil
print(f"Accuracy:  {acc_svm:.4f}")
print(f"Precision: {prec_svm:.4f}")
print(f"Recall:    {rec_svm:.4f}")
print(f"F1-Score:  {f1_svm:.4f}")

# Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Purples', cbar=False)
plt.title('SVM - Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('svm_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: svm_confusion_matrix.png")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))

# ----------------------------------------------------------------------------
# Support Vectors Information
# ----------------------------------------------------------------------------
# Support vectors adalah data points yang menentukan decision boundary
# Semakin sedikit support vectors, semakin "clean" pemisahan kelasnya
# Banyak support vectors bisa indikasi data complex atau noisy

print("\n" + "="*50)
print("SUPPORT VECTORS INFORMATION")
print("="*50)
print(f"Total support vectors: {len(svm.support_)}")
print(f"Support vectors per class: {svm.n_support_}")
print(f"Percentage of training data: {len(svm.support_)/len(X_train)*100:.2f}%")

# Interpretasi:
# - < 30%: Good separation, clean data
# - 30-50%: Moderate complexity
# - > 50%: Complex boundary atau noisy data

if len(svm.support_)/len(X_train) < 0.3:
    print("→ Classes are well-separated with good margin")
elif len(svm.support_)/len(X_train) < 0.5:
    print("→ Moderate complexity in class separation")
else:
    print("→ High complexity or noisy data - consider data cleaning or regularization")

print("="*50)


# ============================================================================
# 4. MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("4. MODEL COMPARISON")
print("="*80)

# Konversi hasil ke DataFrame untuk analisis dan visualisasi
results_df = pd.DataFrame(results).T
print("\nPerformance Metrics Comparison:")
print(results_df.round(4))

# Save hasil ke CSV untuk dokumentasi
results_df.to_csv('model_comparison.csv')
print("\n✓ Results saved to 'model_comparison.csv'")

# ----------------------------------------------------------------------------
# Visualisasi Perbandingan Detail
# ----------------------------------------------------------------------------
# Buat subplot untuk setiap metric
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
    # Bar plot untuk setiap metric
    results_df[metric].plot(kind='bar', ax=ax, color=colors[idx], alpha=0.7)
    ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1.1])  # Range 0-1.1 untuk kasih ruang label
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(results_df.index, rotation=45, ha='right')
    
    # Tambahkan nilai di atas bar
    for i, v in enumerate(results_df[metric]):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison_detailed.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: model_comparison_detailed.png")

# ----------------------------------------------------------------------------
# Visualisasi Perbandingan Overall
# ----------------------------------------------------------------------------
# Semua metric dalam satu grafik
results_df.plot(kind='bar', figsize=(12, 6), width=0.8)
plt.title('Overall Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Model')
plt.ylabel('Score')
plt.legend(loc='lower right')
plt.ylim([0, 1.1])
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison_overall.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: model_comparison_overall.png")

# ----------------------------------------------------------------------------
# Determine Best Model
# ----------------------------------------------------------------------------
# Model terbaik berdasarkan accuracy (bisa ganti dengan metric lain)
best_model_name = results_df['Accuracy'].idxmax()
best_accuracy = results_df['Accuracy'].max()

print("\n" + "="*60)
print(f"🏆 BEST MODEL: {best_model_name}")
print("="*60)
print(f"Accuracy:  {results_df.loc[best_model_name, 'Accuracy']:.4f}")
print(f"Precision: {results_df.loc[best_model_name, 'Precision']:.4f}")
print(f"Recall:    {results_df.loc[best_model_name, 'Recall']:.4f}")
print(f"F1-Score:  {results_df.loc[best_model_name, 'F1-Score']:.4f}")
print("="*60)


# ============================================================================
# 5. HYPERPARAMETER TUNING (Optional - untuk analisis lanjutan)
# ============================================================================
print("\n" + "="*80)
print("5. HYPERPARAMETER TUNING")
print("="*80)
print("NOTE: Tuning bisa memakan waktu lama tergantung parameter grid dan CV folds")
print("Untuk praktikum, tuning 1-2 model saja sudah cukup untuk analisis")

# ----------------------------------------------------------------------------
# 5.1 KNN Tuning
# ----------------------------------------------------------------------------
print("\n5.1 Tuning KNN...")
print("Testing combinations of K, weights, and distance metrics...")

# Grid parameter yang akan dicoba
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],              # Coba berbagai K
    'weights': ['uniform', 'distance'],           # Uniform vs distance-weighted
    'metric': ['euclidean', 'manhattan']          # Distance metrics
}
# Total combinations: 5 × 2 × 2 = 20 models akan di-test

# GridSearchCV: test semua kombinasi dengan cross-validation
grid_knn = GridSearchCV(
    KNeighborsClassifier(),
    param_grid_knn,
    cv=5,               # 5-fold cross-validation
    scoring='accuracy', # Optimize untuk accuracy
    n_jobs=-1           # Use all CPU cores
)
grid_knn.fit(X_train_scaled, y_train)

print(f"✓ Best parameters: {grid_knn.best_params_}")
print(f"  Best CV score: {grid_knn.best_score_:.4f}")

# ----------------------------------------------------------------------------
# 5.2 Decision Tree Tuning
# ----------------------------------------------------------------------------
print("\n5.2 Tuning Decision Tree...")
print("Testing combinations of criterion, depth, and split parameters...")

param_grid_dt = {
    'criterion': ['gini', 'entropy'],             # Splitting criterion
    'max_depth': [3, 5, 7, 10, None],            # Tree depth
    'min_samples_split': [2, 5, 10],             # Min samples to split
    'min_samples_leaf': [1, 2, 5]                # Min samples in leaf
}
# Total combinations: 2 × 5 × 3 × 3 = 90 models

grid_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid_dt,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_dt.fit(X_train, y_train)  # DT tidak butuh scaling

print(f"✓ Best parameters: {grid_dt.best_params_}")
print(f"  Best CV score: {grid_dt.best_score_:.4f}")

# ----------------------------------------------------------------------------
# 5.3 Logistic Regression Tuning
# ----------------------------------------------------------------------------
print("\n5.3 Tuning Logistic Regression...")
print("Testing combinations of C and penalty...")

param_grid_logreg = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],         # Regularization strength
    'penalty': ['l1', 'l2'],                      # Regularization type
    'solver': ['liblinear', 'saga']               # Solver (support l1)
}
# Total combinations: 6 × 2 × 2 = 24 models
# Note: tidak semua solver support semua penalty

grid_logreg = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid_logreg,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_logreg.fit(X_train_scaled, y_train)

print(f"✓ Best parameters: {grid_logreg.best_params_}")
print(f"  Best CV score: {grid_logreg.best_score_:.4f}")

# ----------------------------------------------------------------------------
# 5.4 SVM Tuning
# ----------------------------------------------------------------------------
print("\n5.4 Tuning SVM...")
print("Testing combinations of C, gamma, and kernel...")
print("WARNING: SVM tuning is computationally expensive!")

param_grid_svm = {
    'C': [0.1, 1, 10, 100],                      # Regularization
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1], # Kernel coefficient
    'kernel': ['rbf', 'poly']                     # Kernel type
}
# Total combinations: 4 × 5 × 2 = 40 models
# SVM training lambat, bisa skip kalau waktu terbatas

grid_svm = GridSearchCV(
    SVC(random_state=42),
    param_grid_svm,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_svm.fit(X_train_scaled, y_train)

print(f"✓ Best parameters: {grid_svm.best_params_}")
print(f"  Best CV score: {grid_svm.best_score_:.4f}")

# ----------------------------------------------------------------------------
# Tuned Models Comparison
# ----------------------------------------------------------------------------
print("\n" + "-"*80)
print("TUNED MODELS COMPARISON")
print("-"*80)

# Bandingkan CV score setelah tuning
tuned_results = {
    'KNN': grid_knn.best_score_,
    'Decision Tree': grid_dt.best_score_,
    'Logistic Regression': grid_logreg.best_score_,
    'SVM': grid_svm.best_score_
}

tuned_df = pd.DataFrame(tuned_results.items(), columns=['Model', 'Best CV Score'])
tuned_df = tuned_df.sort_values('Best CV Score', ascending=False)
print(tuned_df)

# Compare sebelum dan sesudah tuning
print("\n" + "-"*80)
print("IMPROVEMENT AFTER TUNING")
print("-"*80)
comparison = pd.DataFrame({
    'Model': results_df.index,
    'Before (Test Acc)': results_df['Accuracy'].values,
    'After (CV Score)': [tuned_results[m] for m in results_df.index]
})
comparison['Improvement'] = comparison['After (CV Score)'] - comparison['Before (Test Acc)']
print(comparison.round(4))

# Visualisasi improvement
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(comparison))
width = 0.35

bars1 = ax.bar(x - width/2, comparison['Before (Test Acc)'], width, label='Before Tuning', alpha=0.7)
bars2 = ax.bar(x + width/2, comparison['After (CV Score)'], width, label='After Tuning', alpha=0.7)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Accuracy Score', fontsize=12)
ax.set_title('Model Performance: Before vs After Hyperparameter Tuning', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.ylim([0, 1.1])
plt.tight_layout()
plt.savefig('tuning_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: tuning_comparison.png")


# ============================================================================
# 6. FINAL SUMMARY & ANALYSIS GUIDE
# ============================================================================
print("\n" + "="*80)
print("6. FINAL SUMMARY & ANALYSIS GUIDE")
print("="*80)

# ----------------------------------------------------------------------------
# Performance Summary
# ----------------------------------------------------------------------------
print("\n📊 PERFORMANCE SUMMARY:")
print(results_df.round(4))

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   - Accuracy:  {results_df.loc[best_model_name, 'Accuracy']:.4f}")
print(f"   - Precision: {results_df.loc[best_model_name, 'Precision']:.4f}")
print(f"   - Recall:    {results_df.loc[best_model_name, 'Recall']:.4f}")
print(f"   - F1-Score:  {results_df.loc[best_model_name, 'F1-Score']:.4f}")

# ----------------------------------------------------------------------------
# Files Generated
# ----------------------------------------------------------------------------
print("\n📁 Generated Files:")
files_list = [
    'target_distribution.png',
    'knn_confusion_matrix.png',
    'knn_k_optimization.png',
    'dt_confusion_matrix.png',
    'dt_feature_importance.png',
    'dt_tree_visualization.png',
    'logreg_confusion_matrix.png',
    'logreg_coefficients.png',
    'svm_confusion_matrix.png',
    'model_comparison.csv',
    'model_comparison_detailed.png',
    'model_comparison_overall.png',
    'tuning_comparison.png'
]
for f in files_list:
    print(f"   - {f}")

# ----------------------------------------------------------------------------
# Analysis Guide untuk Laporan
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("📝 ANALYSIS GUIDE FOR REPORT (DEADLINE 22.00)")
print("="*80)

print("""
STRUKTUR LAPORAN YANG DISARANKAN:

1. DESKRIPSI DATASET
   - Jumlah sampel dan fitur
   - Distribusi target (balanced/imbalanced?)
   - Range dan karakteristik fitur
   - Missing values handling (jika ada)

2. PREPROCESSING
   - Feature scaling: kenapa perlu? algoritma mana yang butuh?
   - Train-test split: 80-20, menggunakan stratify
   - Justifikasi setiap langkah preprocessing

3. MODEL IMPLEMENTATION & RESULTS
   Untuk setiap algoritma, diskusikan:
   
   a) K-NEAREST NEIGHBORS (KNN)
      - Parameter: K=5, distance=euclidean
      - Hasil: Acc={:.4f}, Prec={:.4f}, Rec={:.4f}, F1={:.4f}
      - Optimal K: {} (dari cross-validation)
      - Analisis confusion matrix
      - Kelebihan: sederhana, no training phase, non-parametric
      - Kekurangan: slow prediction, sensitive to scale, curse of dimensionality
      
   b) DECISION TREE
      - Parameter: criterion=entropy, max_depth=5
      - Hasil: Acc={:.4f}, Prec={:.4f}, Rec={:.4f}, F1={:.4f}
      - Feature importance: fitur mana yang paling penting?
      - Analisis struktur pohon (jika di-visualize)
      - Kelebihan: interpretable, no scaling needed, handle non-linear
      - Kekurangan: prone to overfitting, unstable, bias pada imbalanced
      
   c) LOGISTIC REGRESSION
      - Parameter: penalty=l2, C=1.0
      - Hasil: Acc={:.4f}, Prec={:.4f}, Rec={:.4f}, F1={:.4f}
      - Coefficients: fitur dengan pengaruh positif/negatif
      - Analisis decision boundary (linear?)
      - Kelebihan: fast, probabilistic, regularization
      - Kekurangan: linear boundary only, feature engineering needed
      
   d) SUPPORT VECTOR MACHINE (SVM)
      - Parameter: kernel=rbf, C=1.0, gamma=scale
      - Hasil: Acc={:.4f}, Prec={:.4f}, Rec={:.4f}, F1={:.4f}
      - Support vectors: berapa % dari training data?
      - Analisis margin dan boundary complexity
      - Kelebihan: effective high-dim, memory efficient, versatile kernels
      - Kekurangan: slow on large data, sensitive to params, need scaling

4. MODEL COMPARISON
   - Ranking berdasarkan accuracy, precision, recall, F1
   - Grafik perbandingan (sudah di-generate)
   - Model mana yang terbaik? Kenapa?
   - Trade-off antara performa dan kompleksitas

5. HYPERPARAMETER TUNING (Optional tapi bagus untuk nilai)
   - Parameter apa saja yang di-tune?
   - Best parameters untuk setiap model
   - Improvement setelah tuning
   - Grafik before vs after tuning

6. CONFUSION MATRIX ANALYSIS
   - Untuk setiap model, analisis:
     * True Positives (TP): prediksi benar untuk kelas positif
     * True Negatives (TN): prediksi benar untuk kelas negatif
     * False Positives (FP): error Type I
     * False Negatives (FN): error Type II
   - Model mana yang paling banyak FP? FN?
   - Implikasi error untuk use case spesifik

7. KESIMPULAN
   - Model terbaik: {} dengan accuracy {:.4f}
   - Justifikasi pemilihan model (bukan hanya berdasarkan accuracy)
   - Pertimbangan:
     * Performa (accuracy, precision, recall, F1)
     * Complexity (training time, interpretability)
     * Scalability (besar dataset, dimensi fitur)
     * Trade-offs yang acceptable untuk use case
   - Saran improvement untuk future work

8. LEARNING POINTS
   - Insight apa yang didapat?
   - Challenges yang dihadapi?
   - Perbedaan teori vs praktek?

TIPS ANALISIS:
- Jangan hanya report angka, JELASKAN kenapa angka tersebut terjadi
- Hubungkan dengan karakteristik dataset
- Compare trade-offs antar algoritma
- Berikan insight praktis, bukan hanya teori
- Gunakan visualisasi untuk support argumen
""".format(
    results['KNN']['Accuracy'], results['KNN']['Precision'], 
    results['KNN']['Recall'], results['KNN']['F1-Score'],
    
    results['Decision Tree']['Accuracy'], results['Decision Tree']['Precision'],
    results['Decision Tree']['Recall'], results['Decision Tree']['F1-Score'],
    
    results['Logistic Regression']['Accuracy'], results['Logistic Regression']['Precision'],
    results['Logistic Regression']['Recall'], results['Logistic Regression']['F1-Score'],
    
    results['SVM']['Accuracy'], results['SVM']['Precision'],
    results['SVM']['Recall'], results['SVM']['F1-Score'],
    
    best_model_name, best_accuracy
))

# ----------------------------------------------------------------------------
# Quick Reference: Common Issues & Solutions
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("⚠️  COMMON ISSUES & SOLUTIONS")
print("="*80)
print("""
ISSUE: Model accuracy terlalu tinggi (>99%)
SOLUTION: Mungkin data leakage atau overfitting
- Check apakah target bocor ke features
- Reduce model complexity (max_depth, C, dll)
- Use cross-validation untuk verify

ISSUE: Model accuracy terlalu rendah (<50%)
SOLUTION: Underfitting atau data quality issue
- Check data preprocessing
- Increase model complexity
- Feature engineering
- Check class distribution (imbalanced?)

ISSUE: High accuracy tapi low precision
SOLUTION: Many False Positives
- Model terlalu "optimistic" dalam prediksi positif
- Tune threshold atau adjust class weights
- Consider cost of FP in your use case

ISSUE: High accuracy tapi low recall
SOLUTION: Many False Negatives
- Model terlalu "conservative"
- Adjust threshold atau sampling strategy
- Consider cost of FN in your use case

ISSUE: KNN/SVM/LogReg perform badly
SOLUTION: Lupa scaling!
- WAJIB StandardScaler untuk KNN, SVM, LogReg
- Check: mean ≈ 0, std ≈ 1

ISSUE: Decision Tree overfit (100% train acc)
SOLUTION: Tree terlalu dalam
- Set max_depth (try 3, 5, 7)
- Increase min_samples_split
- Increase min_samples_leaf
- Consider pruning

ISSUE: Training sangat lambat
SOLUTION: 
- Reduce dataset size (sampling)
- Reduce parameter grid (tuning)
- Use simpler kernel (SVM: linear instead of rbf)
- Reduce CV folds (5 -> 3)
""")

print("\n" + "="*80)
print("✅ PRAKTIKUM COMPLETE!")
print("="*80)
print("""
CHECKLIST SEBELUM SUBMIT:
□ Kode running tanpa error
□ Semua 4 algoritma implemented
□ Evaluation metrics calculated
□ Confusion matrix untuk semua model
□ Model comparison table & grafik
□ File output tersimpan dengan benar
□ (Optional) Hyperparameter tuning
□ (Deadline 22.00) Analisis lengkap & interpretasi

GOOD LUCK! 🚀

Need help? Review:
- Cheat sheet (IF3170_Praktikum2_CheatSheet.md)
- Code comments (throughout this file)
- Generated visualizations
""")
print("="*80)
