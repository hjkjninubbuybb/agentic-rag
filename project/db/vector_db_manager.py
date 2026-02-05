# project/db/vector_db_manager.py

# ============================================================
# 导入部分
# ============================================================

# [本项目] config
# 来源：project/config.py
# 用法：读取全局配置（如数据库路径、模型名称、稀疏向量字段名）。
import config

# [第三方库] langchain_huggingface
# 来源：LangChain 的 HuggingFace 集成
# 用法：HuggingFaceEmbeddings 用于加载强大的开源模型（如 bge-m3, all-mpnet-base-v2）。
# 作用：负责生成"稠密向量" (Dense Vector)，用于理解句子的意思。
from langchain_huggingface import HuggingFaceEmbeddings

# [第三方库] langchain_qdrant
# 来源：LangChain 的 Qdrant 集成
# 用法：
# - QdrantVectorStore: 包装器，让 LangChain 能操作 Qdrant。
# - FastEmbedSparse: 专门用于生成"稀疏向量" (Sparse Vector)，实现关键词搜索。
# - RetrievalMode: 枚举类，用于指定检索模式（这里我们用 HYBRID 混合模式）。
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

# [第三方库] qdrant_client
# 来源：Qdrant 官方客户端
# 用法：负责底层的数据库连接、建表、删除等操作。
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


# ============================================================
# 类定义: VectorDbManager (支持混合检索)
# ============================================================
class VectorDbManager:
    # [类型提示] 定义私有成员变量的类型，方便 IDE 提示
    __client: QdrantClient
    __dense_embeddings: HuggingFaceEmbeddings
    __sparse_embeddings: FastEmbedSparse

    def __init__(self):
        """
        初始化：连接数据库，并同时加载两套模型（稠密+稀疏）。
        """
        # 1. 连接 Qdrant 数据库 (本地文件模式)
        self.__client = QdrantClient(path=config.QDRANT_DB_PATH)

        # 2. 加载稠密向量模型 (语义理解)
        # model_name 在 config.py 中配置 (如 "sentence-transformers/all-mpnet-base-v2")
        self.__dense_embeddings = HuggingFaceEmbeddings(model_name=config.DENSE_MODEL)

        # 3. 加载稀疏向量模型 (关键词匹配)
        # 这里的模型通常是 "Qdrant/bm25"，非常轻量，用于弥补语义搜索不够精确的缺点
        self.__sparse_embeddings = FastEmbedSparse(model_name=config.SPARSE_MODEL)

    # ------------------------------------------------------------
    # 创建集合 (Create Table)
    # ------------------------------------------------------------
    def create_collection(self, collection_name):
        """
        创建一个支持混合检索的 Qdrant 集合。
        """
        # 检查是否已存在
        if not self.__client.collection_exists(collection_name):
            print(f"Creating collection: {collection_name}...")

            # [核心逻辑] 创建集合配置
            self.__client.create_collection(
                collection_name=collection_name,

                # 配置 1: 稠密向量 (Dense)
                # size: 自动获取模型输出维度 (比如 768)
                # distance: 使用余弦相似度 (Cosine)
                vectors_config=qmodels.VectorParams(
                    size=len(self.__dense_embeddings.embed_query("test")),
                    distance=qmodels.Distance.COSINE
                ),

                # 配置 2: 稀疏向量 (Sparse)
                # 这就是混合检索的关键！为 BM25 关键词索引预留位置。
                sparse_vectors_config={
                    config.SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()
                },
            )
            print(f"✓ Collection created: {collection_name}")
        else:
            print(f"✓ Collection already exists: {collection_name}")

    # ------------------------------------------------------------
    # 删除集合
    # ------------------------------------------------------------
    def delete_collection(self, collection_name):
        """
        删除指定的集合 (清空数据)。
        """
        try:
            if self.__client.collection_exists(collection_name):
                print(f"Removing existing Qdrant collection: {collection_name}")
                self.__client.delete_collection(collection_name)
        except Exception as e:
            print(f"Warning: could not delete collection {collection_name}: {e}")

    # ------------------------------------------------------------
    # 获取 LangChain 包装器
    # ------------------------------------------------------------
    def get_collection(self, collection_name) -> QdrantVectorStore:
        """
        返回一个配置好"混合检索模式"的 VectorStore 对象。
        """
        try:
            # [关键配置] 启用 Hybrid Search
            return QdrantVectorStore(
                client=self.__client,
                collection_name=collection_name,

                # 传入两个模型：一个看意思，一个看关键词
                embedding=self.__dense_embeddings,  # 稠密模型
                sparse_embedding=self.__sparse_embeddings,  # 稀疏模型

                # [核心] 设置模式为 HYBRID (混合)
                # 搜索时会同时计算：语义分数 + 关键词匹配分数
                retrieval_mode=RetrievalMode.HYBRID,

                # 告诉它稀疏向量存在哪个字段里 (对应 create_collection 里的配置)
                sparse_vector_name=config.SPARSE_VECTOR_NAME
            )
        except Exception as e:
            print(f"Unable to get collection {collection_name}: {e}")
            # 如果出错，这里可能需要 raise e 或者返回 None，避免程序继续带病运行
            raise e