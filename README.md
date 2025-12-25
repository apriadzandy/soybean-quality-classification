<h1 align="center">Klasifikasi Kualitas Kacang Kedelai Menggunakan Deep Learning</h1>

<p align="center">
  <img src="Assets/Tampilan%20Dasboard.jpeg" width="700"/>
</p>

---

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk membangun sistem **klasifikasi kualitas kacang kedelai** berbasis **Deep Learning menggunakan data citra**.  
Sistem ini mampu mengklasifikasikan citra kacang kedelai ke dalam **lima kelas kualitas**, yaitu:

- Broken soybeans  
- Immature soybeans  
- Intact soybeans  
- Skin-damaged soybeans  
- Spotted soybeans  

Pada proyek ini dilakukan perbandingan performa antara **model CNN non-pretrained** dan **dua model pretrained (transfer learning)**.  
Hasil model kemudian diintegrasikan ke dalam **aplikasi website sederhana menggunakan Streamlit** yang dijalankan secara lokal.

---

## 📂 Dataset
- Dataset yang di gunakan dalam proyek ini di ambil dari dataset publik https://www.kaggle.com/datasets/warcoder/soyabean-seeds
- deskripsi :
  
- Jenis data: **Data citra (image dataset)**
- Jumlah data: **5.000+ citra**
- Jumlah kelas: **5 kelas kualitas kacang kedelai**
- Pembagian data:
  - Training: **70%**
  - Validation: **15%**
  - Testing: **15%**
- Pembagian data dilakukan secara **stratified** untuk menjaga keseimbangan distribusi kelas.

---

## 🧑‍💻 Preprocessing Data
Tahapan preprocessing yang dilakukan meliputi:
1. Resize citra menjadi **224 × 224 pixel**
2. One-hot encoding pada label kelas
3. Data augmentation (khusus data training):
   - Random horizontal flip  
   - Random rotation  
   - Random zoom  
4. Normalisasi data:
   - **CNN Scratch**: Rescaling (1./255)
   - **ResNet50 & MobileNetV2**: preprocess_input sesuai arsitektur pretrained
5. Optimalisasi pipeline data menggunakan **tf.data** dengan prefetch untuk mempercepat proses training

---

## 🤖 Model yang Digunakan

### 1️⃣ CNN Scratch (Non-Pretrained)
Model CNN dibangun dari awal tanpa menggunakan bobot pretrained.  
Model ini digunakan sebagai **baseline** untuk mengevaluasi kemampuan jaringan saraf konvolusional sederhana pada dataset kualitas kacang kedelai.

### 2️⃣ ResNet50 (Transfer Learning)
Model **ResNet50 pretrained ImageNet** digunakan sebagai feature extractor.  
Pelatihan dilakukan dalam dua tahap:
- Freeze base model dan melatih classifier head
- Fine-tuning dengan membuka sebagian layer terakhir menggunakan learning rate kecil

### 3️⃣ MobileNetV2 (Transfer Learning)
Model **MobileNetV2 pretrained ImageNet** digunakan sebagai model ringan dan efisien.  
Pendekatan freeze dan fine-tuning diterapkan untuk menyesuaikan fitur pretrained dengan dataset.

---

## 📊 Hasil Evaluasi dan Analisis Perbandingan Model
### 📑 Classification Report (Ringkasan)

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-score (Macro) |
|------|----------|-------------------|----------------|------------------|
| CNN Scratch (Non-Pretrained) | 0.83 | 0.82 | 0.82 | 0.82 |
| ResNet50 (Transfer Learning) | 0.92 | 0.92 | 0.91 | 0.91 |
| MobileNetV2 (Transfer Learning) | 0.80 | 0.82 | 0.80 | 0.80 |


### Tabel Analisis Perbandingan Model

| Nama Model | Akurasi | Hasil Analisis |
|-----------|---------|----------------|
| CNN Scratch (Non-Pretrained) | 83% | Model baseline dengan performa cukup baik, namun masih kesulitan membedakan kelas dengan karakteristik visual yang mirip, khususnya *broken soybeans*. |
| ResNet50 (Transfer Learning) | 92% | Model terbaik dan paling stabil. Fitur pretrained ImageNet dan fine-tuning mampu meningkatkan pemisahan antar kelas secara signifikan. |
| MobileNetV2 (Transfer Learning) | 80% | Model ringan dan efisien, namun performanya lebih rendah dibanding ResNet50, terutama pada kelas *immature* dan *intact soybeans*. |

**Kesimpulan:**  
Model **ResNet50** memberikan performa terbaik untuk klasifikasi kualitas kacang kedelai berdasarkan hasil eksperimen yang dilakukan.

---

## 📉 Confusion Matrix
Di bawah ini adalah confusion matrix untuk ketiga model.

| **CNN Stracth** | **Resnet50** | **MobileNetV2** |
|---------|---------|-------------------|
| ![Confusion Matrix CNN Stracth](https://github.com/apriadzandy/soybean-quality-classification/blob/main/Assets/Confusion%20Matrix%20CNN%20Stracth.png) | ![Confusion Matrix Resnet50](https://github.com/apriadzandy/soybean-quality-classification/blob/main/Assets/Confusion%20Matrix%20Resnet50.png) | ![Confusion Matrix MobileNetV2](https://github.com/apriadzandy/soybean-quality-classification/blob/main/Assets/Confusion%20Matrix%20MobileNetv2.png) |
---

## 🖥️ Sistem Website Streamlit
Aplikasi Streamlit digunakan sebagai sistem prediksi berbasis web yang dijalankan secara lokal.
![Tampilan Streamlit](Assets/Tampilan%20Dasboard.jpeg)

### Fitur Website
- Upload citra kacang kedelai
- Pemilihan model (CNN Scratch, ResNet50, atau MobileNetV2)
- Menampilkan hasil prediksi kelas kualitas dan confidence

---

## ⚙️ Panduan Menjalankan Sistem Secara Lokal

### 1. Instalasi Dependensi
```bash
  pip install -r requirements.txt
```
### 2. Menjalankan Aplikasi Streamlit
```bash
  streamlit run app.py
```
---

## Catatan
- Dikarenakan Ukuran model resnet50 cukup besar maka dapat di download dari link berikut
- Link Model Lengkap : https://drive.google.com/drive/folders/1Dko7ZnYCDs7g2zcxKseYQlMLFtRZ_a3G?usp=sharing

👤 Biodata

Nama: Apriadzandy Putra

Program Studi: Teknik Informatika

Fakultas: Teknik

Universitas: Universitas Muhammadiyah Malang



