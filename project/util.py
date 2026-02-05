# project/util.py

# ============================================================
# 导入部分
# ============================================================

# [Python标准库] os
# 用法：这里主要用于设置环境变量，禁止 Tokenizers 库的并行警告
import os

# [本项目] config
# 来源：project/config.py
# 用法：读取全局配置，比如 Markdown 文件的默认输出路径 (MARKDOWN_DIR)
import config

# [第三方库] pymupdf (别名 fitz) / pymupdf4llm
# 来源：PyMuPDF 库
# 用法：
# - pymupdf: 强大的 PDF 处理库，用于打开和读取 PDF。
# - pymupdf4llm: 专门为大模型优化的转换工具。它能把 PDF 里的表格、标题、段落智能转换成 Markdown 格式，而不是乱糟糟的纯文本。
import pymupdf.layout
import pymupdf4llm

# [Python标准库] pathlib.Path
# 用法：面向对象的文件路径处理库（比 os.path 好用）。
from pathlib import Path

# [Python标准库] glob
# 用法：用于文件查找，比如找到文件夹下所有的 "*.pdf"。
import glob

# [配置] 禁用 Tokenizers 并行
# 当你在多进程环境中使用 HuggingFace 的 tokenizers 库时，如果不关掉这个，经常会报死锁警告。
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# 函数: 单个 PDF 转 Markdown
# ============================================================
def pdf_to_markdown(pdf_path, output_dir):
    """
    将单个 PDF 文件转换为 Markdown 文件并保存。
    """
    # [第三方库] 打开 PDF 文件
    doc = pymupdf.open(pdf_path)

    # [第三方库] 核心转换逻辑
    # pymupdf4llm.to_markdown 会分析页面布局，尽量保留表格结构和标题层级。
    # - ignore_images=True: 我们只关注文本内容，忽略图片（为了节省 Token）。
    # - write_images=False: 不把图片提取存盘。
    md = pymupdf4llm.to_markdown(
        doc,
        header=False,
        footer=False,
        page_separators=True,  # 保留分页符，方便以后回溯页码
        ignore_images=True,
        write_images=False,
        image_path=None
    )

    # [Python逻辑] 编码清洗
    # 这一步是为了防止 PDF 中包含一些生僻的 Unicode 字符导致写入文件时报错。
    # 'surrogatepass' 允许处理代理对字符，'ignore' 忽略无法解码的垃圾字符。
    md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')

    # [Python逻辑] 构造输出路径
    # 例如: output_dir/report.pdf -> output_dir/report
    output_path = Path(output_dir) / Path(doc.name).stem

    # [Python逻辑] 写入文件
    # .with_suffix(".md") 确保后缀是 .md
    Path(output_path).with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))


# ============================================================
# 函数: 批量转换
# ============================================================
def pdfs_to_markdowns(path_pattern, overwrite: bool = False):
    """
    扫描指定路径下的所有 PDF 并批量转换。

    Args:
        path_pattern: 文件匹配模式，例如 "data/*.pdf"
        overwrite: 是否覆盖已存在的 Markdown 文件 (默认 False，跳过已存在的以节省时间)
    """
    # [本项目] 从配置读取输出目录，并确保目录存在
    output_dir = Path(config.MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # [Python标准库] glob.glob 遍历匹配的文件
    # map(Path, ...) 把文件名字符串转成 Path 对象
    for pdf_path in map(Path, glob.glob(path_pattern)):

        # 预测目标文件路径
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")

        # [逻辑] 增量更新检查
        # 如果文件已存在且不强制覆盖 (overwrite=False)，直接跳过
        if overwrite or not md_path.exists():
            print(f"🔄 Converting: {pdf_path.name} -> {md_path.name} ...")
            pdf_to_markdown(pdf_path, output_dir)
        else:
            print(f"⏩ Skipping (already exists): {md_path.name}")