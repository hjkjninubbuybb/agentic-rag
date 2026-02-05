# project/config.py

# [Python标准库] 操作系统接口，用于读取环境变量
import os

# ============================================================
# [关键修复] 设置 HuggingFace 国内镜像站
# 作用：强制让 Python 从国内镜像下载模型，解决连接超时 (ReadTimeoutError)
# 注意：这行代码最好放在文件最前面
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# [第三方库] python-dotenv，用于加载 .env 文件中的变量到环境变量中
from dotenv import load_dotenv

# [第三方库] 执行加载：会自动寻找项目根目录下的 .env 文件
# 如果找到了，SILICONFLOW_API_KEY 就会被加载到 os.environ 中
load_dotenv()

# --- 目录配置 (Directory Configuration) ---
# [配置] Markdown 文件存放路径
MARKDOWN_DIR = "markdown_docs"
# [配置] 父文档存储路径 (JSON文件)
PARENT_STORE_PATH = "parent_store"
# [配置] 向量数据库路径 (Qdrant本地文件)
QDRANT_DB_PATH = "qdrant_db"

# --- Qdrant 集合配置 ---
# [配置] 子文档集合名称
CHILD_COLLECTION = "document_child_chunks"
# [配置] 稀疏向量字段名 (用于关键词搜索)
SPARSE_VECTOR_NAME = "sparse"

# --- 嵌入模型配置 (Embedding Configuration) ---
# [配置] 密集向量模型 (语义搜索)
# [第三方库] 使用 HuggingFace 的模型，会在本地运行
# 添加了镜像配置后，这里下载就会飞快了
DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"
# [配置] 稀疏向量模型 (关键词搜索)
SPARSE_MODEL = "Qdrant/bm25"

# --- LLM 配置 (SiliconFlow) ---

# [配置] 硅基流动的模型名称
LLM_MODEL = "deepseek-ai/DeepSeek-V3"

# [配置] 从环境变量中读取 API Key
# 如果 .env 没配置好，第二个参数是报错提示
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    raise ValueError("❌ 未找到 SILICONFLOW_API_KEY，请检查你的 .env 文件！")

# [配置] 硅基流动的 Base URL (OpenAI 兼容接口地址)
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# [配置] 温度设为 0 (让回答最严谨、不发散)
LLM_TEMPERATURE = 0

# --- 文本切分配置 (Text Splitter Configuration) ---
# [配置] 子文档大小 (500字符)，用于检索
CHILD_CHUNK_SIZE = 500
# [配置] 子文档重叠 (100字符)，防止上下文切断
CHILD_CHUNK_OVERLAP = 100
# [配置] 父文档最小/最大字符数，用于给 AI 提供上下文
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 10000
# [配置] Markdown 标题切分规则
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]