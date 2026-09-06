# 即夢影片 CLI

此 CLI 不是操作即夢網頁版，而是使用火山方舟影片生成 API。即夢網頁版目前沒有供個人帳號使用的官方 CLI；火山方舟是正式可程式化的接入方式。

## 1. 開通與設定

1. 在火山引擎控制台開通「火山方舟」。
2. 建立 API Key。
3. 開通一個支援圖生影片的 Seedance 模型或建立推理接入點。
4. 在專案 `.env` 加入：

```dotenv
ARK_API_KEY=你的方舟_API_Key
ARK_VIDEO_MODEL=你的模型_ID_或推理接入點_ID
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_VIDEO_POLL_SECONDS=10
```

不要把 API Key 貼到對話、程式碼或 Git。

## 2. 查看指令

```powershell
python scripts/jimeng_cli.py --help
python scripts/jimeng_cli.py generate --help
```

## 3. 圖片生成影片

```powershell
python scripts/jimeng_cli.py generate `
  --image "core/assets/podcast_studio_john_mary.png" `
  --prompt "John gently waves and blinks. Mary smiles and makes a small welcoming gesture. Keep the microphone stable. Slow cinematic camera push-in. Preserve character identity, outfits, composition, and 3D cartoon style. No morphing or close-up." `
  --resolution 720p `
  --duration 5 `
  --ratio "16:9" `
  --output "workspace/podcast_studio_john_mary.mp4"
```

本機圖片會轉為 Data URL 後提交。也可以把 `--image` 改成公開 HTTPS 圖片 URL。

## 4. 只提交、不等待

```powershell
python scripts/jimeng_cli.py generate `
  --image "core/assets/podcast_studio_john_mary.png" `
  --prompt "Subtle natural character motion. Keep the microphone stable." `
  --no-wait
```

記下輸出的 `task_id`，之後可以查詢：

```powershell
python scripts/jimeng_cli.py status TASK_ID
```

或等待完成並下載：

```powershell
python scripts/jimeng_cli.py wait TASK_ID `
  --output "workspace/podcast_studio_john_mary.mp4"
```

## 5. 常用選項

- `--model`：臨時覆蓋 `.env` 的 `ARK_VIDEO_MODEL`。
- `--camera-fixed`：要求固定鏡頭。
- `--watermark`：要求平台浮水印，預設關閉。
- `--resolution`：`480p`、`720p` 或 `1080p`。
- `--duration`：影片秒數，實際可用值取決於所選模型。
- `--ratio`：例如 `16:9`、`9:16` 或 `1:1`，實際支援值取決於模型。

影片生成會消耗火山方舟額度。影片下載 URL 可能有時效，完成後應立即下載。
