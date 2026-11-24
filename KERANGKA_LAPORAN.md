# Kerangka Laporan ATM: Dampak LLM pada Code Review (sesuai bobot penilaian)

> Isi dengan angka dan kata-kata Anda sendiri. Bagian [ISI] harus berasal dari hasil eksperimen Anda.

## 1. Related Work & Knowledge Gap (10%)
- Ringkas 4–5 studi (Tufano 2024; Xiao 2024 Copilot; Hao 2024; Kudrjavets 2022; Davila 2024), 1 kalimat tiap studi.
- Gap: belum ada analisis per fase PR. Kritik Anda: pelabelan hanya via kata kunci "GPT".
- Novelty ATM Anda: [ISI, mis. model embedding berbeda, Oracle Vector Search, koreksi Bonferroni, LLM sebagai annotator].

## 2. Masalah & Research Question (10%)
- RQ1: Apakah PR GPT-assisted lebih cepat di-merge? (H0: median sama)
- RQ2: Fase mana yang berubah? (submission/review/waiting/change)
- RQ3: Untuk tugas apa GPT dipakai?
- [ISI] RQ tambahan milik Anda (opsional).

## 3. Data (10%)
- Sumber: GitHub REST API, query [ISI], tanggal ambil [ISI].
- Filter: stars ≥ 10, merged, ≥ 2 partisipan. Jumlah: awal [N] -> akhir [N].
- Tabel distribusi assisted vs non-assisted. Statistik deskriptif fitur.

## 4. Metode (20%)
Diagram alur: Collect -> Label (regex + LLM + validasi manual) -> Fitur -> Matching -> Statistik.
- Label: akurasi regex vs manual [ISI]%, Cohen's kappa [ISI].
- Matching: Manhattan (min-max). Implementasi Oracle: `VECTOR_DISTANCE(..., MANHATTAN)`, `VECTOR(4,FLOAT32)`, HNSW index.
- Hubungan dengan materi OA: vector = representasi fitur; distance function; vector index; RAG (opsional untuk klasifikasi tugas).
- Statistik: Mann-Whitney U + rank-biserial, regresi OLS log1p, Bonferroni untuk 4 fase.
- Asumsi regresi: cek VIF dan residual (ISI).

## 5. Hasil & Analisis (20%)
- Tabel median TTM assisted vs non (Anda vs paper: 9 jam vs 23 jam).
- Tabel regresi + R². Tabel fase + p-value + effect size.
- Distribusi tugas (enhancement/bugfix/doc/...).
- Analisis: apakah hasil Anda searah dengan paper? Jika tidak, mengapa? (selection bias, confounding, ukuran sampel).
- Sensitivitas: ulangi dengan label regex saja vs label LLM.

## 6. Kesimpulan (10%)
- Jawab tiap RQ dalam 1–2 kalimat berisi angka. Keterbatasan. Kerja lanjutan.
- Catatan: asosiasi, bukan kausalitas.

## 7. Lain-lain (20%)
- Link GitHub (kode, dataset, prompt LLM, requirements.txt, README cara menjalankan).
- Screenshot: kursus OA 100%, eksekusi kode, query Oracle, tabel hasil.
- Submit tautan kode dan laporan di MS Teams.

## 8. Cara Menjalankan Aman
- Atur token GitHub sebagai variabel lingkungan `GITHUB_TOKEN`; jangan tulis token atau password ke dalam kode/repo.
- Jalankan demo: `python atm_phase_oracle.py demo`.
- Untuk Oracle, teruskan password melalui parameter `pwd` dari environment/secret management yang aman.
