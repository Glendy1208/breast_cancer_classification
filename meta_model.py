import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle

# Memuat dataset
df = pd.read_csv('breast_cancer_knn.csv')
df = df.drop(['Unnamed: 0'], axis=1)

# Memisahkan fitur dan label
feature_columns = ["Clump_thickness","Uniformity_of_cell_size","Uniformity_of_cell_shape",
                   "Marginal_adhesion","Single_epithelial_cell_size", "Bare_nuclei",
                   "Bland_chromatin","Normal_nucleoli","Mitoses"]
X = df[feature_columns].values
y = df['Class'].values

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Membuat daftar estimators untuk StackingClassifier
estimators = [('knn' + str(i // 3), KNeighborsClassifier(n_neighbors=i)) for i in range(3, 63, 3)]

# Menginisialisasi StackingClassifier dengan GaussianNB sebagai final estimator
clf = StackingClassifier(
    estimators=estimators, final_estimator=GaussianNB()
)

# Melatih StackingClassifier
clf.fit(X_train, y_train)

# Mengevaluasi akurasi masing-masing model KNN pada data uji
for name, estimator in estimators:
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of {name}: {accuracy:.4f}")

# Mengevaluasi performa model keseluruhan
overall_accuracy = clf.score(X_test, y_test)
print(f'\nOverall Stacking Classifier Performance:')
print(f'Accuracy: {overall_accuracy:.4f}')

# Melakukan prediksi untuk data baru X_new
X_new = [[1, 1, 1, 1, 1, 1, 1, 1, 1]]
prediction = clf.predict(X_new)
print(f'\nPrediction for X_new: {prediction[0]}')

# Melakukan prediksi untuk data baru X_new menggunakan setiap estimator
predictions = {}
for name, estimator in estimators:
    pred = estimator.predict(X_new)[0]
    predictions[name] = pred
    print(f"Prediction of {name}: {pred}")

# Menghitung jumlah prediksi untuk kelas 2 dan kelas 4
count_class_2 = sum(1 for pred in predictions.values() if pred == 2)
count_class_4 = sum(1 for pred in predictions.values() if pred == 4)

print(f'\nNumber of estimators predicting class 2: {count_class_2}')
print(f'Number of estimators predicting class 4: {count_class_4}')

# Melatih ulang model (jika perlu)
clf.fit(X_train, y_train)

# Menyimpan model ke dalam file 'stacking_classifier.pkl'
with open('stacking_classifier.pkl', 'wb') as file:
    pickle.dump(clf, file)