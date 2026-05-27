# Key LivePortrait Local

Project này chạy hoàn toàn trên máy local.

## Kiến Trúc

- `server.js`: Node.js web server, serve giao diện và nhận request `/api/generate`.
- `public/`: giao diện upload ảnh/video và nhận video output.
- `python/motion_avatar_pipeline.py`: pipeline local, gọi LivePortrait và hậu xử lý video.
- `LivePortrait/`: repo LivePortrait đã clone vào root project.
- `.venv310/`: Python 3.10 virtualenv dùng để chạy pipeline.
- `.env`: cấu hình đường dẫn local cho Node server.

Luồng chạy:

```text
Browser UI -> Node /api/generate -> Python pipeline -> LivePortrait local GPU -> output.mp4
```

## Yêu Cầu

- Windows có NVIDIA GPU.
- Python 3.10.
- Node.js.
- LivePortrait weights nằm trong `LivePortrait/pretrained_weights`.

Trên máy hiện tại, project đã được cấu hình trong `.env`:

```text
LIVEPORTRAIT_REPO=C:\Users\NhanLe\Documents\Project\key_liveportrait_local\key_liveportrait\LivePortrait
PYTHON_BIN=C:\Users\NhanLe\Documents\Project\key_liveportrait_local\key_liveportrait\.venv310\Scripts\python.exe
PORT=3000
```

## Chạy Project

Mở PowerShell tại root project:

```powershell
cd C:\Users\NhanLe\Documents\Project\key_liveportrait_local\key_liveportrait
npm start
```

Sau đó mở trình duyệt:

```text
http://localhost:3000
```

Dừng server:

```text
Ctrl + C
```

## Kiểm Tra Server

Mở:

```text
http://localhost:3000/api/health
```

Kết quả tốt sẽ có:

```json
{
  "ok": true,
  "mode": "local-node-to-liveportrait",
  "liveportraitConfigured": true
}
```

## Cài Đặt Lại Từ Đầu

Nếu cần setup lại trên máy khác:

```powershell
npm install
py -3.10 -m venv .venv310
.\.venv310\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv310\Scripts\python.exe -m pip install -r requirements.txt
.\.venv310\Scripts\python.exe -m pip install -r LivePortrait\requirements.txt
.\.venv310\Scripts\python.exe -m pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

Tải weights LivePortrait:

```powershell
cd LivePortrait
$env:PYTHONUTF8='1'
..\.venv310\Scripts\hf.exe download KlingTeam/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
cd ..
```

Tạo `.env` từ `.env.example`, sau đó sửa path cho đúng máy.

## Log Khi Generate

Khi chạy `npm start`, log Python sẽ hiện trực tiếp trong terminal, gồm:

- Param pipeline: `grain_strength`, `motion_blur_alpha`, `brightness`, `contrast`, `gamma`, `saturation`, `sharpen_amount`, `driving_multiplier`.
- Ảnh source được chọn.
- Log LivePortrait.
- Lỗi nếu pipeline fail.

## Lỗi Thường Gặp

`scipy` đòi build source hoặc báo thiếu Fortran:

- Đang dùng sai Python, thường là Python 3.13.
- Dùng `.venv310` Python 3.10 như `.env` đã cấu hình.

`liveportraitConfigured: false`:

- Sai `LIVEPORTRAIT_REPO` trong `.env`.
- Hoặc folder `LivePortrait` không có `inference.py`.

Không thấy log `print()`:

- Phải chạy bằng terminal `npm start`.
- Log sẽ hiện trong terminal đang chạy server, không phải trong browser console.

Port `3000` vẫn chạy sau khi tắt VS Code:

- Nghĩa là Node đang chạy background.
- Tìm và tắt process:

```powershell
netstat -ano | Select-String ':3000'
Stop-Process -Id <PID> -Force
```

## File Không Nên Commit

Đã ignore các thư mục/file local nặng:

- `.env`
- `.venv310/`
- `LivePortrait/`
- `local_jobs/`
- `uploads/`
- `node_modules/`
