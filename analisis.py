# Step 1: Pengumpulan Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Membaca data dari file CSV
data = pd.read_csv('data_penjualan.csv')

# Step 2: Pembersihan Data
# Memeriksa nilai yang hilang
print(data.isnull().sum())

# Mengatasi nilai yang hilang (jika ada)
data.fillna(method='ffill', inplace=True)

# Memeriksa dan menghapus duplikasi
data.drop_duplicates(inplace=True)

# Memeriksa data yang tidak konsisten atau salah
print(data.describe())
print(data['Jenis Kelamin'].unique())
print(data['Jenis Barang'].unique())

# Step 3: Transformasi Data
# Mengubah format tanggal
data['Tanggal Pembelian'] = pd.to_datetime(data['Tanggal Pembelian'])

# Step 4: Eksplorasi Data (EDA)
# Statistik deskriptif
print(data.describe())

# Visualisasi data
plt.figure(figsize=(10,6))
sns.countplot(x='Jenis Barang', hue='Jenis Kelamin', data=data)
plt.title('Distribusi Jenis Barang Berdasarkan Jenis Kelamin')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='Jenis Barang', y='Jumlah Barang', data=data)
plt.title('Boxplot Jumlah Barang per Jenis Barang')
plt.show()

# Step 5: Pemodelan Data
# Misalkan kita ingin memprediksi jenis barang berdasarkan fitur lainnya
# Mengkodekan variabel kategori
data_encoded = pd.get_dummies(data, columns=['Jenis Kelamin', 'Jenis Barang'])

# Memisahkan fitur dan target
X = data_encoded.drop(columns=['Nomor Faktur', 'Tanggal Pembelian', 'Jumlah Barang'])
y = data_encoded['Jenis Barang']
# Memisahkan data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pemilihan model dan pelatihan model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Validasi dan Tuning Model
# Evaluasi model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Step 7: Interpretasi dan Penyajian Hasil
# Menyajikan hasil analisis
feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)

# Step 8: Deploy dan Monitoring
# Membuat API untuk model (contoh menggunakan Flask)
# from flask import Flask, request, jsonify
# app = Flask(__name__)
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     prediction = model.predict([data])
#     return jsonify({'prediction': prediction.tolist()})
# if __name__ == '__main__':
#     app.run(debug=True)

# Step 9: Maintenance dan Iterasi
# Monitoring performa model secara berkala
# Retraining jika diperlukan
