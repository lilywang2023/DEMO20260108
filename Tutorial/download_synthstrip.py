import os
import tarfile
from pathlib import Path
import gdown

# ===========================
# 配置
# ===========================
DATA_NAME = "synthstrip_data_v1.5_2d"
GOOGLE_DRIVE_ID = "1IvVw2JxD690P8v4mc6FMi4hx6xlx9hV6"
TAR_FILENAME = f"{DATA_NAME}.tar"


def main():
    tar_path = Path(TAR_FILENAME)
    extract_dir = Path(DATA_NAME)

    # 1. 下载 .tar 文件（如果不存在）
    if not tar_path.exists():
        print(f"📥 正在从 Google Drive 下载 {TAR_FILENAME} ...")
        print("🔗 共享链接: https://drive.google.com/uc?id=" + GOOGLE_DRIVE_ID)

        try:
            # 使用 gdown.download() 直接下载，自动显示进度条
            gdown.download(
                id=GOOGLE_DRIVE_ID,
                output=str(tar_path),
                quiet=False,  # 显示进度条（默认 True 是静默）
                fuzzy=False  # 精确匹配 ID
            )
            print("\n✅ 下载完成!")
        except Exception as e:
            print(f"\n❌ 下载失败: {e}")
            return
    else:
        print(f"📁 {TAR_FILENAME} 已存在，跳过下载")

    # 2. 解压 .tar 文件（如果未解压）
    if not extract_dir.exists():
        print(f"📦 正在解压 {TAR_FILENAME} ...")
        try:
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=".")
            print("✅ 解压完成!")
        except Exception as e:
            print(f"❌ 解压失败: {e}")
            return
    else:
        print(f"📁 目录 '{DATA_NAME}' 已存在，跳过解压")

    # 3. 列出关键内容
    print(f"\n📂 数据目录内容 ({DATA_NAME}):")
    if extract_dir.exists():
        for item in sorted(extract_dir.iterdir()):
            if item.is_file():
                print(f"  - {item.name} ({item.stat().st_size / 1024:.1f} KB)")
    else:
        print("  ⚠️ 目录不存在")


if __name__ == "__main__":
    main()