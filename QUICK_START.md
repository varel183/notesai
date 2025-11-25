# QUICK START GUIDE - IF3170 PRAKTIKUM 2
**Rabu, 26 November 2025**

## ⏰ Timeline
- **09.00**: Deadline 1 - Kode Program Lengkap
- **22.00**: Deadline 2 - Analisis Kode dan Hasil

## 📋 Checklist Deadline 09.00

### ✅ Yang Harus Diserahkan:
1. Kode program lengkap (.py atau .ipynb)
2. Running tanpa error
3. Implementasi 4 algoritma:
   - K-Nearest Neighbors (KNN)
   - Decision Tree Learning (DTL)
   - Logistic Regression
   - Support Vector Machine (SVM)
4. Evaluation metrics:
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
5. Model comparison (tabel)

### 🚀 Cara Pakai Template:

#### Step 1: Setup (5 menit)
```python
# 1. Buka praktikum_template.py
# 2. Install dependencies jika perlu:
pip install numpy pandas matplotlib seaborn scikit-learn
```

#### Step 2: Load Data (10 menit)
```python
# Cari bagian ini di template (line ~50):
# TODO: GANTI BAGIAN INI DENGAN DATA KAMU

# Uncomment dan sesuaikan:
df = pd.read_csv('your_data.csv')
X = df.drop('target_column', axis=1)
y = df['target_column']

# Hapus bagian iris dataset demo
```

#### Step 3: Run! (30 menit)
```bash
python praktikum_template.py
```
atau run di Jupyter Notebook/Google Colab

#### Step 4: Check Output (5 menit)
File yang harus ada:
- ✅ target_distribution.png
- ✅ knn_confusion_matrix.png
- ✅ dt_confusion_matrix.png
- ✅ logreg_confusion_matrix.png
- ✅ svm_confusion_matrix.png
- ✅ model_comparison.csv
- ✅ model_comparison_detailed.png

#### Step 5: Submit (5 menit)
Submit kode + semua output files sebelum jam 09.00!

---

## 📋 Checklist Deadline 22.00

### ✅ Analisis yang Harus Ada:

#### 1. Deskripsi Dataset (10 menit)
- Jumlah sampel dan fitur
- Distribusi kelas (balanced/imbalanced)
- Karakteristik data
```
Contoh:
"Dataset terdiri dari 150 sampel dengan 4 fitur numerik.
Distribusi kelas balanced: setosa (50), versicolor (50), virginica (50).
Tidak ada missing values."
```

#### 2. Preprocessing (10 menit)
- Kenapa pakai StandardScaler?
  → "KNN, LogReg, dan SVM sensitive terhadap skala fitur"
- Kenapa stratify split?
  → "Maintain proporsi kelas di train dan test"

#### 3. Hasil Tiap Model (40 menit)

**Template per Model:**
```
[NAMA MODEL]
- Parameters: [list parameter yang digunakan]
- Results:
  * Accuracy:  X.XXXX
  * Precision: X.XXXX
  * Recall:    X.XXXX
  * F1-Score:  X.XXXX
  
- Analisis Confusion Matrix:
  * TP: [angka] - [interpretasi]
  * FP: [angka] - [interpretasi]
  * TN: [angka] - [interpretasi]
  * FN: [angka] - [interpretasi]
  
- Kelebihan pada dataset ini:
  [2-3 poin spesifik]
  
- Kekurangan pada dataset ini:
  [2-3 poin spesifik]
```

#### 4. Perbandingan Model (20 menit)

**Buat tabel:**
| Model | Accuracy | Precision | Recall | F1 | Rank |
|-------|----------|-----------|--------|----|----|
| KNN | X.XX | X.XX | X.XX | X.XX | 2 |
| DT  | X.XX | X.XX | X.XX | X.XX | 3 |
| LR  | X.XX | X.XX | X.XX | X.XX | 4 |
| SVM | X.XX | X.XX | X.XX | X.XX | 1 |

**Analisis:**
- Model terbaik: [nama] karena [alasan]
- Model terburuk: [nama] karena [alasan]
- Trade-off yang diamati: [diskusi]

#### 5. Hyperparameter Tuning (Optional - 30 menit)

**Jika ada waktu:**
- Parameter apa yang di-tune?
- Best parameters untuk tiap model
- Improvement setelah tuning
- Grafik before vs after

#### 6. Kesimpulan (10 menit)

**Must include:**
- Model terbaik + justifikasi
- Insight dari eksperimen
- Rekomendasi untuk use case tertentu

```
Template:
"Berdasarkan eksperimen, [MODEL] memberikan performa terbaik
dengan accuracy [X.XX]. Model ini dipilih karena:
1. [Alasan 1]
2. [Alasan 2]
3. [Alasan 3]

Untuk dataset ini, trade-off antara [X] dan [Y] acceptable
karena [justifikasi]."
```

---

## 🎯 Tips Cepat

### Kalau Waktu Terbatas:
1. **Skip hyperparameter tuning** - pakai default parameters OK
2. **Fokus ke analisis 4 model dasar**
3. **Grafik comparison sudah auto-generated** - tinggal interpretasi

### Kalau Ada Waktu Lebih:
1. ✅ Hyperparameter tuning (grid search)
2. ✅ Feature importance analysis (DT)
3. ✅ K optimization visualization (KNN)
4. ✅ Coefficients analysis (LogReg)

### Common Mistakes to Avoid:
- ❌ Lupa scaling untuk KNN/LogReg/SVM
- ❌ Tidak stratify split (penting untuk imbalanced data)
- ❌ Comparison tidak fair (beda preprocessing)
- ❌ Hanya report angka tanpa interpretasi
- ❌ Lupa save output files

---

## 📊 Interpretasi Metrics

### Accuracy
- **High (>90%)**: Model sangat baik ATAU overfitting
- **Medium (70-90%)**: Normal untuk most datasets
- **Low (<70%)**: Underfitting atau data quality issue

### Precision vs Recall
- **High Precision, Low Recall**: Conservative model (sedikit FP, banyak FN)
  → Use case: spam detection (rather miss spam than block real email)
  
- **Low Precision, High Recall**: Aggressive model (banyak FP, sedikit FN)
  → Use case: disease screening (rather false alarm than miss disease)
  
- **Both High**: Perfect! 🎯
- **Both Low**: Model buruk atau data jelek

### F1-Score
- Harmonic mean of Precision & Recall
- Good untuk imbalanced dataset
- Single number to judge model

---

## 🆘 Troubleshooting

### Error: "ValueError: Input contains NaN"
**Solution:** Ada missing values
```python
# Check
print(X.isnull().sum())

# Fix
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
```

### Error: "ValueError: Unknown label type"
**Solution:** Target perlu di-encode
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```

### Warning: "ConvergenceWarning"
**Solution:** LogReg belum converge
```python
# Increase max_iter
LogisticRegression(max_iter=2000)
```

### Model Accuracy = 100% on training
**Solution:** OVERFITTING!
- Reduce max_depth (DT)
- Increase regularization (decrease C for LogReg/SVM)
- Increase K (KNN)

---

## 📚 Quick Formula Reference

### KNN
```
Distance = √(Σ(xi - yi)²)  # Euclidean
```

### Decision Tree
```
Entropy(S) = -Σ(pi × log2(pi))
IG = Entropy(parent) - Σ(weighted_entropy(children))
```

### Logistic Regression
```
σ(z) = 1/(1 + e^(-z))
Cost = -(1/m)Σ[y·log(h(x)) + (1-y)·log(1-h(x))]
```

### SVM
```
Margin = 2/||w||
Maximize Margin ⟺ Minimize ||w||²/2
```

### Evaluation
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## ✅ Final Checklist Sebelum Submit

### Deadline 09.00:
- [ ] Kode running tanpa error
- [ ] Semua 4 algoritma implemented
- [ ] Output files generated
- [ ] Model comparison table ada
- [ ] Submit tepat waktu

### Deadline 22.00:
- [ ] Deskripsi dataset
- [ ] Preprocessing explanation
- [ ] Hasil per model (accuracy, confusion matrix)
- [ ] Analisis kelebihan/kekurangan tiap model
- [ ] Model comparison & interpretation
- [ ] Kesimpulan & justifikasi model terbaik
- [ ] Submit tepat waktu

---

## 🚀 Final Tips

1. **Prioritas:**
   - Kode working > Perfect tuning
   - Clear analysis > Complex analysis
   - Submit on time > Perfect score

2. **Dokumentasi:**
   - Comment your code!
   - Explain your choices
   - Show your understanding

3. **Time Management:**
   - 08.00-09.00: Final check & submit kode
   - 09.00-18.00: Analisis & interpretasi
   - 18.00-21.00: Write report
   - 21.00-22.00: Final review & submit

4. **Jangan Lupa:**
   - Isi pembagian kelompok (deadline besok 09.00!)
   - Tidak boleh pakai AI (per instruksi praktikum)
   - Boleh buka cheat sheet & dokumentasi

---

**GOOD LUCK! 🎓**

Files yang sudah disiapkan:
1. ✅ IF3170_Praktikum2_CheatSheet.md - Formula & konsep lengkap
2. ✅ praktikum_template.py - Kode siap pakai dengan comment detail
3. ✅ QUICK_START.md - Guide ini

Semua ada di `/mnt/user-data/outputs/`
