import os
import gdown

# 配置
WEIGHTS_FILE = 'weights.h5'
GOOGLE_DRIVE_ID = '19ayDE-otx2kTcGzEhz_pv9MjnJkpX638'

# 下载前检查：若文件不存在，则下载
if not os.path.exists(WEIGHTS_FILE):
    print(f"📥 正在下载模型权重到 '{WEIGHTS_FILE}'...")
    gdown.download(id=GOOGLE_DRIVE_ID, output=WEIGHTS_FILE, quiet=False)
else:
    print(f"✅ 权重文件 '{WEIGHTS_FILE}' 已存在，跳过下载。")
