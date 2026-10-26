# MultiFrame-LPR: End-to-End Super-Resolution & Transformer for Low-Resolution License Plate Recognition

Dự án nghiên cứu luận văn thạc sĩ về bài toán **Nhận dạng biển số xe độ phân giải thấp từ chuỗi đa khung hình (Multi-Frame Low-Resolution License Plate Recognition)**.

Repository này được tạo ra để phục vụ quá trình trao đổi, thảo luận kỹ thuật và nhận hỗ trợ/tư vấn từ đồng nghiệp nhằm tối ưu hóa kiến trúc mô hình và nâng cao độ chính xác nhận dạng.

---

## 📌 1. Bối cảnh & Mục tiêu bài toán

- **Đầu vào (Input):** Chuỗi $F = 5$ khung hình liên tiếp chụp từ video giám sát giao thông thực tế. Các khung hình bị suy biến nặng (độ phân giải thấp, mờ do chuyển động, biến dạng góc nghiêng, nhiễu nén).
- **Đầu ra (Output):** Chuỗi ký tự biển số chuẩn xác (7 ký tự, định dạng biển số xe máy/ô tô theo chuẩn Mercosur/Việt Nam).
- **Mục tiêu trọng tâm:** 
  - Khôi phục chi tiết ký tự bằng mạng siêu phân giải (Super-Resolution).
  - Tích hợp không gian - thời gian (Spatio-Temporal Fusion) từ đa khung hình.
  - Cải thiện độ chính xác phân biệt các ký tự có hình thái topo gần giống nhau (`6` $\leftrightarrow$ `8`, `8` $\leftrightarrow$ `B`, `1` $\leftrightarrow$ `3`, `H` $\leftrightarrow$ `M`).
  - Đưa độ chính xác Exact Match vượt ngưỡng **$\ge 80.5\%$**.

---

## 🏗️ 2. Kiến trúc Mô hình (Model Architecture)

Hệ thống được thiết kế theo 2 luồng tiếp cận chính:

```
[5 LR Frames: B x 5 x 3 x H x W]
              │
              ▼
   ┌──────────────────────┐
   │    StackedSRNet      │ (Frame-Parallel Residual Blocks + 2x Upsampling)
   └──────────────────────┘
              │
              ▼
   [5 SR Frames: B x 5 x 3 x 2H x 2W]
              │
              ▼
   ┌──────────────────────┐
   │    Spatial Trans.    │ (STN: Căn chỉnh phối cảnh & góc nghiêng)
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │   ResNet Backbone    │ (Trích xuất đặc trưng hình thái)
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Attention Fusion     │ (Hợp nhất thông tin thời gian từ 5 frames)
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Transformer Encoder  │ (Mô hình hóa phụ thuộc ngữ cảnh chuỗi ký tự)
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ CTC Classifier Head  │ ──► HardCharacterCTCLoss (Focal + Angular Margin)
   └──────────────────────┘
              │
              ▼
   [Chuỗi 7 ký tự biển số]
```

### Flow 2: `MultiFrameSRResTran` (`src/models/sr_restran.py`)
- **Super-Resolution (SR Module):** `StackedSRNet` với 8 khối Residual Block xử lý song song từng frame, upsample tỉ lệ $2\times$ giúp tái tạo sắc nét biên cạnh ký tự.
- **OCR Module (`ResTranOCR`):**
  - **STN (Spatial Transformer Network):** Tự động nắn thẳng góc nghiêng và bù trừ chuyển động giữa các frame.
  - **Backbone:** ResNet-34 tiền huấn luyện.
  - **Multi-Head Attention Fusion:** Hợp nhất đặc trưng không gian - thời gian giữa 5 frames.
  - **Transformer Encoder:** 3 lớp Transformer Encoder ($8$ heads, feed-forward dimension $2048$) trích xuất phụ thuộc ngữ cảnh chuỗi.
  - **CTC Loss Head:** Dự đoán chuỗi ký tự không cần gán nhãn từng ký tự căn chỉnh thời gian.

### Flow 3: `MultiFrameSRViT` (`src/models/sr_vit.py`)
- Kết hợp `StackedSRNet` cùng Vision Transformer (ViT) Backbone trích xuất token patches trực tiếp trên không gian đặc trưng siêu phân giải.

---

## 🎯 3. Đóng góp & Kỹ thuật huấn luyện đặc trưng

### a. Two-Stage Decoupled Training Pipeline (`train_decoupled.py`)
Huấn luyện End-to-End từ đầu dễ gây mất ổn định (gradient của CTC loss phá vỡ khả năng tạo ảnh nét của mạng SR). Chúng tôi triển khai quy trình Decoupled 2 giai đoạn:
1. **Stage 1 (Frozen SR Adaptation):** Đóng băng toàn bộ trọng số mạng SR (đã học tiền nghiệm khôi phục ảnh ở mức $\sim 26.11\text{ dB}$ PSNR), chỉ huấn luyện mạng OCR thích nghi với miền ảnh SR tái tạo.
2. **Stage 2 (Joint Fine-tuning with Differential LR):** Mở khóa toàn bộ mạng nhưng áp dụng tốc độ học vi sai:
   - $\text{LR}_{\text{SR}} = 3.0 \times 10^{-6}$ (rất nhỏ để giữ ổn định trọng số SR)
   - $\text{LR}_{\text{OCR}} = 3.0 \times 10^{-5}$
   - Kết hợp Cosine Annealing Scheduler và Grad Norm Clipping ($1.0$).

### b. Hard-Character CTC Loss with Prototype Margin (`src/training/ctc_loss.py`)
Giải quyết hiện tượng nhầm lẫn giữa các ký tự có đường nét hình học gần tương tự:
$$\mathcal{L} = \mathcal{L}_{\text{CTC-Focal}} + \lambda_{\text{margin}} \cdot \mathcal{L}_{\text{margin}}$$
- **Focal CTC:** Tăng trọng số phạt đối với các chuỗi chứa ký tự dễ nhầm lẫn hoặc độ bất định căn chỉnh cao.
- **Prototype Angular Margin Regularization ($\mathcal{L}_{\text{margin}}$):** Ép khoảng cách góc giữa các vector trọng số prototype $(w_a, w_b)$ của các cặp ký tự dễ nhầm lẫn (`6` $\leftrightarrow$ `8`, `8` $\leftrightarrow$ `B`, `1` $\leftrightarrow$ `3`, v.v.) trong tầng phân loại CTC phải lớn hơn một ngưỡng margin an toàn $m$, ngăn hiện tượng sập không gian đặc trưng.

---

## 📊 4. Kết quả Thực nghiệm Hiện tại (Scenario-B Validation)

| Chỉ số Đánh giá (Metrics) | Baseline (CRNN) | Baseline (ResTran) | **Mô hình đề xuất (Flow 2)** |
| :--- | :---: | :---: | :---: |
| **Exact Match Accuracy (Seq Acc)** | ~70.20% | ~73.10% | **76.68%** |
| **Character Error Rate (CER)** | ~8.45% | ~7.60% | **6.49%** |
| **Normalized Edit Distance Acc** | ~90.80% | ~91.90% | **93.51%** |
| **Macro F1-Score** | - | - | **92.48%** |
| **Micro F1-Score** | - | - | **93.84%** |

*(Kết quả thu được sau đợt Full-Dataset Overnight Training 15 epochs trên 100% dữ liệu 38.002 tracks, đánh giá trên 999 tracks của tập Scenario-B).*

---

## 💡 5. Các điểm nghẽn & Câu hỏi cần xin ý kiến / hỗ trợ

Rất mong nhận được sự góp ý và thảo luận từ các anh/chị về các định hướng sau:

1. **Chiến lược đẩy Sequence Accuracy từ ~76.7% lên $\ge 80.5\%$:**
   - Liệu có nên bổ sung cơ chế Masked Language Modeling (MLM) hoặc 2D Attention Decoder thay cho Greedy CTC Decoding thuần túy?
   - Kỹ thuật Data Augmentation nào hiệu quả nhất cho chuỗi video độ phân giải thấp (CutMix, Elastic Distortion, hay Frame Drop Simulation)?
2. **Giải quyết triệt để 5 cặp nhầm lẫn phổ biến nhất hiện nay:**
   - `8` nhầm thành `6` (và ngược lại)
   - `H` nhầm thành `M`
   - `1` nhầm thành `3`
   - `8` nhầm thành `0`
   - Có nên áp dụng ArcFace/Sub-center CosFace trực tiếp lên các feature tokens trước khi đưa vào classifier head?
3. **Cơ chế Temporal Fusion đa khung hình:**
   - Hiện tại Attention Fusion áp dụng đơn giản giữa các frame. Liệu kiến trúc Temporal Convolution (TCN) hoặc Deformable 3D Attention có giúp bắt chuyển động của ký tự tốt hơn không?
4. **Cân bằng đa nhiệm (Multi-task Objective Balance):**
   - Tỉ lệ tối ưu giữa $\lambda_{\text{sr}}$ (Perceptual + Pixel Loss) và $\mathcal{L}_{\text{CTC}}$ để mạng SR không bị "quá khớp" vào việc làm mờ ảnh giả tạo để giảm CTC loss.

---

## 🚀 6. Hướng dẫn Chạy (Quick Start)

### Cài đặt môi trường
```bash
git clone https://github.com/QuanDo610/LRLPR-2026-Support.git
cd LRLPR-2026-Support

# Tạo môi trường ảo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install albumentations
```

### Chạy Đánh giá (Evaluation)
Để kiểm tra độ chính xác mô hình Flow 2 trên tập dữ liệu kiểm định:
```bash
python evaluate.py \
  --model sr_restran \
  --weights results/decoupled_restran_best.pth \
  --output-json results/evaluation_metrics.json
```

### Chạy Huấn luyện Two-Stage Decoupled
```bash
# Giai đoạn huấn luyện Decoupled (Frozen SR + Joint Fine-tuning)
python train_decoupled.py \
  --model restran \
  --frozen-epochs 5 \
  --joint-epochs 10 \
  --batch-size 48 \
  --lr-ocr-stage1 1.5e-4 \
  --lr-ocr-stage2 3.0e-5 \
  --lr-sr-stage2 3.0e-6 \
  --lambda-margin 4.0
```

Hoặc chạy toàn bộ script overnight tự động (tích hợp `caffeinate` trên macOS):
```bash
bash scripts/run_overnight_full_train.sh
```

---

## 📂 7. Cấu trúc thư mục chính

```
.
├── configs/                # Cấu hình siêu tham số (config.py)
├── src/
│   ├── data/               # Multi-frame Dataset & Data Augmentations
│   ├── models/             # Kiến trúc mạng (sr.py, restran.py, sr_restran.py, sr_vit.py)
│   ├── training/           # CTC Loss, Margin Loss, Trainer loop
│   └── utils/              # Các hàm bổ trợ (metrics, seeds, visualizer)
├── scripts/                # Scripts chạy Kaggle, Overnight, Benchmark, Latency
├── train_decoupled.py      # Quy trình huấn luyện tách rời 2 giai đoạn (Flow 2)
├── train_ocr_hard.py       # Huấn luyện OCR chuyên sâu chống nhầm lẫn
├── evaluate.py             # Đánh giá toàn diện (Exact Match, CER, Confusion Matrix)
├── TRAINING_PROGRESS_REPORT.md # Báo cáo nhật ký huấn luyện
└── README.md
```

---
*Mọi ý kiến đóng góp, xin liên hệ qua Issue hoặc Pull Request trên repository này.*
