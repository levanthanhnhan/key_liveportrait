# Avatar Motion Web

Web app Node.js + frontend hiện đại để upload nhiều ảnh mặt, upload driving video, chỉnh thông số hậu kỳ, rồi gọi Python + LivePortrait để tạo video output.

## 1. Cài Node dependencies

```bash
npm install
```

## 2. Cài Python dependencies

```bash
pip install opencv-python numpy
```

## 3. Cài LivePortrait

Tạo thư mục `external` rồi clone LivePortrait:

```bash
mkdir -p external
git clone https://github.com/KwaiVGI/LivePortrait.git external/LivePortrait
cd external/LivePortrait
pip install -r requirements.txt
```

Tải pretrained weights theo hướng dẫn chính thức của repo LivePortrait.

## 4. Chạy web

```bash
npm start
```

Mở:

```text
http://localhost:3000
```

## 5. Cấu hình môi trường nếu cần

```bash
PYTHON_BIN=python3 \
LIVEPORTRAIT_REPO=/absolute/path/to/LivePortrait \
npm start
```

## Ghi chú

- Dùng cho avatar có consent.
- Output mặc định có disclosure text `AI-generated avatar`.
- Nếu muốn production, nên chuyển job sang queue như BullMQ/Redis thay vì giữ request HTTP quá lâu.
