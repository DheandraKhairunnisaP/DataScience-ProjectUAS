<div align="center">

# FUSI DATA PENGINDERAAN JAUH DAN PARAMETER CUACA IN SITU UNTUK PREDIKSI KERENTANAN PANGAN TERHADAP ANOMALI IKLIM

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Scikit--Learn%20%7C%20PyTorch-orange)](https://pytorch.org/)
[![SDG 2](https://img.shields.io/badge/SDG-2%3A%20Zero%20Hunger-D39F2A)](https://sdgs.un.org/goals/goal2)
[![SDG 13](https://img.shields.io/badge/SDG-13%3A%20Climate%20Action-48783E)](https://sdgs.un.org/goals/goal13)

</div>

Repositori ini memuat berkas proposal, perancangan pembelajaran mesin, serta dokumentasi eksperimen model prediksi hasil panen dan deteksi kerentanan pangan berbasis anomali agroklimat. Seluruh pengembangan disusun mengacu pada siklus L-M-O-P (*Learning Problem, Model Design, Optimization, Predict & Evaluate*).

## DAFTAR PUSTAKA DAN BERKAS DOKUMEN

Struktur direktori dokumentasi pada folder `docs/` dirinci sebagai berikut:

* [docs/09_LAMPIRAN_B.md](docs/09_LAMPIRAN_B.md)
  Formulir pengajuan ide checkpoint 1 yang memuat rumusan target regresi, daftar tautan dataset, rencana tiga model, mitigasi risiko etika, dan referensi ilmiah.
* [docs/09_minggu9.md](docs/09_minggu9.md)
  Panduan capaian tugas dan kriteria checkpoint tahap awal untuk perumusan masalah pembelajaran.
* [docs/Laporan_Proyek_ML_LMOP.md](docs/Laporan_Proyek_ML_LMOP.md)
  Draf utama template laporan komprehensif dari Bab 1 hingga Bab 10 yang mencakup seluruh alur siklus hidup proyek.

---

## RINGKASAN PROYEK

Proyek riset ini bertujuan memitigasi dampak kekeringan dan pergeseran musim tanam melalui peramalan hasil panen tanaman pangan per satuan luas lahan. Pendekatan komputasi memanfaatkan tiga tingkatan model secara bertahap:

* Baseline Model
  *Linear Regression* untuk mengevaluasi korelasi linear dasar antara fitur agroklimat dengan angka produktivitas panen.
* Advanced Model
  *Random Forest Regressor* dan *XGBoost Regressor* untuk memetakan ambang batas cuaca ekstrem serta hubungan non-linear antarvariabel.
* Deep Learning Model
  *Multilayer Perceptron* dengan arsitektur multi-layer teratur menggunakan aktivasi ReLU, *Batch Normalization*, serta *Dropout* guna mereduksi risiko *overfitting* data tabular.
