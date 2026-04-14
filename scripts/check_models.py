import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("--- 2026 專案可用模型清單 ---")
# 新版 SDK 的模型清單物件屬性為 name
for m in client.models.list():
    print(f"ID: {m.name}")