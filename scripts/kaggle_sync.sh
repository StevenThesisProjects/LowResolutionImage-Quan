#!/bin/bash

# Exit on error
set -e

echo "🚀 Bắt đầu đồng bộ Code lên Kaggle Dataset (dotrongminhquan/lrlpr-codebase)..."

TMP_DIR="kaggle_push/lrlpr-codebase"
rm -rf kaggle_push
mkdir -p "$TMP_DIR/source"

# Copy mã nguồn quan trọng (Bỏ qua .venv, .git, __pycache__)
echo "📦 Đang copy mã nguồn..."
rsync -a --exclude='.venv' --exclude='.git' --exclude='__pycache__' --exclude='kaggle_push' --exclude='*.pt' \
  src configs scripts train*.py evaluate.py "$TMP_DIR/source/"


# Copy data (Explicitly resolve symlinks)
echo "📦 Đang copy val_tracks.json..."
mkdir -p "$TMP_DIR/source/data"
cp data/val_tracks.json "$TMP_DIR/source/data/" 2>/dev/null || cp dataset/data/val_tracks.json "$TMP_DIR/source/data/" 2>/dev/null || true

# Copy weights đã warm-start nếu có
if [ -d "results" ]; then
    mkdir -p "$TMP_DIR/source/results"
    cp results/sr_net_best.pth "$TMP_DIR/source/results/" 2>/dev/null || true
    cp results/sr_restran_flow2_plus10_best.pth "$TMP_DIR/source/results/" 2>/dev/null || true
    cp results/sr_vit_flow3_best.pth "$TMP_DIR/source/results/" 2>/dev/null || true
fi

# Zip toàn bộ thư mục source thành codebase.zip (Để Kaggle auto-unzip giữ nguyên cấu trúc)
echo "🗜️ Đang nén toàn bộ project thành codebase.zip..."
cd "$TMP_DIR/source"
zip -r ../codebase.zip ./* > /dev/null
cd ..
# Xóa thư mục source vì chỉ cần upload file zip
rm -rf source

# Khởi tạo dataset metadata cho thư mục chỉ chứa codebase.zip
# ../../.venv/bin/kaggle datasets init -p . > /dev/null 2>&1 || true

# Tạo dataset-metadata.json
cat <<EOF > dataset-metadata.json
{
  "title": "LRLPR Codebase",
  "id": "dotrongminhquan/lrlpr-codebase",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
EOF

# Thử tạo mới, nếu đã tồn tại thì tạo phiên bản mới
echo "☁️ Đang upload lên Kaggle Datasets..."
echo "☁️ Đang cập nhật phiên bản mới..."
../../.venv/bin/kaggle datasets version -p . -m "Auto-update" || ../../.venv/bin/kaggle datasets create -p .


cd ../..

echo "🎉 Xong! Dataset codebase đã được đẩy lên Kaggle."
