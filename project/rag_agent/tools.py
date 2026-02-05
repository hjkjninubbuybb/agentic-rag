# project/rag_agent/tools.py

# ============================================================
# 导入部分
# ============================================================

# [Python标准库] typing.List
# 用法：用于类型提示，表示这是一个列表
from typing import List

# [第三方库] langchain_core.tools.tool
# 来源：LangChain 核心库
# 用法：这是一个装饰器 (@tool)。
# 作用：它能把一个普通的 Python 函数瞬间变成一个 LLM 可以识别和调用的"工具对象"。
# 它会自动提取函数的 docstring (文档字符串) 作为工具的描述，告诉 LLM 这个工具是干嘛的。
from langchain_core.tools import tool

# [本项目] db.parent_store_manager
# 来源：project/db/parent_store_manager.py
# 用法：导入父文档管理器，用于根据 ID 读取存硬盘上的大段文本。
from db.parent_store_manager import ParentStoreManager


# ============================================================
# 类定义: ToolFactory (工具工厂)
# ============================================================
class ToolFactory:
    """
    [类功能] 负责创建和组装 Agent 所需的工具列表。
    为什么要写成类？因为我们需要注入 collection (向量库连接) 和 parent_store_manager (文件存储连接)。
    """

    def __init__(self, collection):
        # [本项目] collection 是从 VectorDbManager 传进来的 Qdrant 集合对象
        self.collection = collection
        # [本项目] 实例化父文档管理器
        self.parent_store_manager = ParentStoreManager()

    # ------------------------------------------------------------
    # 内部函数：搜索子文档 (Search Child Chunks)
    # ------------------------------------------------------------
    def _search_child_chunks(self, query: str, limit: int) -> str:
        """Search for the top K most relevant child chunks.

        Args:
            query: Search query string
            limit: Maximum number of results to return
        """
        try:
            # [第三方库] collection.similarity_search(...)
            # 来源：LangChain Qdrant 集成
            # 用法：执行向量相似度搜索。
            # 参数：
            # - query: 用户的问题（会自动转成向量）。
            # - k: 返回几条结果。
            # - score_threshold: 相似度阈值（0.7），太不相关的不要。
            results = self.collection.similarity_search(query, k=limit, score_threshold=0.7)

            # [逻辑] 如果没查到
            if not results:
                return "NO_RELEVANT_CHUNKS"

            # [Python逻辑] 格式化输出
            # 把查到的 Document 对象列表转换成一个清晰的字符串，方便 LLM 阅读。
            return "\n\n".join([
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            ])

        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"

    # ------------------------------------------------------------
    # 内部函数：获取父文档 (Retrieve Parent Chunks)
    # ------------------------------------------------------------
    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """Retrieve full parent chunks by their IDs.

        Args:
            parent_id: Parent chunk ID to retrieve
        """
        try:
            # [本项目] self.parent_store_manager.load_content(...)
            # 来源：project/db/parent_store_manager.py
            # 用法：去硬盘上读取那个 JSON 文件，获取完整的大段内容。
            parent = self.parent_store_manager.load_content(parent_id)

            # [逻辑] 如果文件找不到
            if not parent:
                return "NO_PARENT_DOCUMENT"

            # [Python逻辑] 格式化输出
            return (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"

    # ------------------------------------------------------------
    # 公开方法：创建工具列表
    # ------------------------------------------------------------
    def create_tools(self) -> List:
        """Create and return the list of tools."""

        # [第三方库] tool("工具名称")(函数)
        # 来源：LangChain
        # 用法：这里我们手动把上面的"私有方法"（self._xxx）包装成了 LangChain 工具。
        # 为什么要这样做？因为 @tool 装饰器通常用于普通函数，
        # 而在类方法中，我们需要绑定 self 上下文，所以用这种 wrap 方式更灵活。

        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        # 强制重写描述，确保 LLM 能看到准确的 docstring
        search_tool.description = "Search for the top K most relevant child chunks."

        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        retrieve_tool.description = "Retrieve full parent chunks by their IDs."

        # 返回列表，这个列表会被传递给 graph.py 里的 llm.bind_tools()
        return [search_tool, retrieve_tool]