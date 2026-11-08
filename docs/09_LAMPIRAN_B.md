## LAMPIRAN B. FORMULIR PENGAJUAN IDE (DIKUMPULKAN PADA CHECKPOINT 1)

| Komponen | Isian Mahasiswa |
|---|---|
| Nama / NIM | Dheandra Khairunnisa Pambudi / 244311009 <br/> Dzakwan Daris Fakhruddin / 244311011 |
| Judul ide | Fusi Data Penginderaan Jauh dan Parameter Cuaca In Situ untuk Prediksi Kerentanan Pangan Terhadap Anomali Iklim |
| SDG dan target yang dipilih | Proyek ini berfokus utama pada *Goal 2 Zero Hunger* dengan target untuk menjamin sistem produksi pangan yang berkelanjutan dan memperkuat kapasitas adaptasi terhadap perubahan iklim serta cuaca ekstrem, didukung oleh *Goal 13 Climate Action* khususnya target terkait penguatan ketahanan terhadap bahaya bencana iklim. |
| Masalah nyata yang diangkat | Pergeseran musim dan kemarau berkepanjangan memicu tingginya risiko gagal panen komoditas pangan pokok, sehingga ketiadaan sistem peramalan dini berbasis iklim menghambat langkah antisipasi petani dan pembuat kebijakan dalam mendistribusikan air, pupuk, serta cadangan logistik pangan. |
| Tugas ML (Target dan jenis tugas) | - Supervised Learning berupa Regresi (Regression) <br/> - Nilai estimasi hasil panen tanaman pangan (crop yield) per satuan luas lahan dengan satuan hektogram per hektar (hg/ha) berdasarkan variabel iklim, curah hujan, suhu, dan input agronomis |
| Objective dan metrik utama | Model dioptimasi untuk meminimalkan fungsi kerugian Mean Squared Error atau Root Mean Squared Error, sedangkan performa akhir dievaluasi menggunakan metrik Root Mean Squared Error, Mean Absolute Error, dan nilai R-squared untuk mengukur persentase variansi hasil panen yang berhasil dijelaskan. |
| Dataset (sumber, URL, ukuran, jenis data) | - [Kaggle Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset) <br/> - [Agriculture Crop Yield Dataset](https://www.kaggle.com/datasets/samuelotiattakorah/agriculture-crop-yield) <br/> - [Smart Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/miadul/smart-crop-yield-predication-dataset). |
| Rencana tiga model (baseline / advanced / deep learning) | - Linear Regression sebagai pembanding dasar korelasi linear <br/><br/> - Random Forest Regressor atau XGBoost Regressor untuk menangkap pola non linear dan ambang batas anomali cuaca <br/><br/> - deep learning berupa Multilayer Perceptron minimal dua lapisan tersembunyi yang memanfaatkan fungsi aktivasi ReLU serta teknik regularisasi Dropout dan Batch Normalization. |
| Risiko dan pertimbangan etika (bias, privasi) | Risiko utama terletak pada potensi bias representasi wilayah akibat kelengkapan pencatatan stasiun cuaca yang timpang serta risiko kesalahan prediksi hasil panen yang dapat memengaruhi alokasi bantuan, sementara aspek privasi tetap aman karena data yang diolah merupakan agregat statistik makro tanpa informasi identitas pribadi. |
| Referensi awal (minimal 2) | [Khaki, S., dan Wang, L. (2019). Crop yield prediction using deep neural networks. Frontiers in Plant Science, 10, 621.](https://doi.org/10.3389/fpls.2019.00621) <br/><br/> [Van Klompenburg, T., Kassahun, A., dan Catal, C. (2020). Crop yield prediction using machine learning: A systematic literature review. Computers and Electronics in Agriculture, 177, 105709.](https://doi.org/10.1016/j.compag.2020.105709) <br/><br/> [Goodfellow, I., Bengio, Y., dan Courville, A. (2016). Deep Learning. MIT Press.](http://repo.darmajaya.ac.id/4189/1/Deep%20learning_%20adaptive%20computation%20and%20machine%20learning%20%28%20PDFDrive%20%29.pdf) |

Kriteria persetujuan dosen:
- [ ] Ide berasal dari mahasiswa dan tidak duplikat
- [ ] Selaras jelas dengan satu SDG dan targetnya
- [ ] Dapat dirumuskan sebagai masalah ML dengan target yang jelas
- [ ] Dataset tersedia, legal/etis digunakan, dan cukup besar
- [ ] Layak untuk tiga model termasuk deep learning dalam 6 minggu
