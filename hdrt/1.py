import os
import time
from google import genai
from google.genai import types
from PIL import Image

API_KEY = "YOURAPIKEY"

IMAGE_FOLDER = "/home/mochiao/code/Python_notes_2025/hdrt/Img/0"

MODEL_ID = "gemini-3-flash-preview" 

def get_hex_files(start_hex="0x31", end_hex="0x41"):
    """生成 0x31.png 到 0x41.png 的文件名列表"""
    files = []
    start = int(start_hex, 16)
    end = int(end_hex, 16)
    for i in range(start, end + 1):
        files.append(f"0x{i:x}.png")
    return files

def main():
    client = genai.Client(api_key=API_KEY)
    
    target_files = get_hex_files()
    print(f"正在使用模型: {MODEL_ID}")
    print(f"准备处理 {len(target_files)} 张图片...\n")

    for filename in target_files:
        filepath = os.path.join(IMAGE_FOLDER, filename)
        
        if not os.path.exists(filepath):
            print(f"[跳过] 文件不存在: {filepath}")
            continue
            
        try:
            image = Image.open(filepath)
            
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[
                    image, 
                    "Identify this single character. Output ONLY the character."
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0, 
                )
            )
            
            char_result = response.text.strip() if response.text else "Error"
            
            print(f"图片: {filename} -> 识别结果: {char_result}")
            
            time.sleep(0.5)

        except Exception as e:
            print(f"[错误] 处理 {filename} 失败: {e}")

if __name__ == "__main__":
    main()
