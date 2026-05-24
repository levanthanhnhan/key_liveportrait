# Avatar Motion Web - Render Node Proxy to Colab GPU

This version is designed for free/demo deployment:

- Render runs the Node.js web app.
- Google Colab runs LivePortrait on GPU.
- ngrok exposes the Colab FastAPI URL.
- Render forwards uploaded files to Colab and streams back the output video.

## Render setup

Build Command:

```bash
npm install && pip install -r requirements.txt
```

Start Command:

```bash
npm start
```

Environment Variable:

```text
PYTHON_API_URL=https://YOUR_NGROK_URL.ngrok-free.app
```

## Important

This Render app does not run LivePortrait locally. If your logs show `Python exited`, you are using the old server.js.
