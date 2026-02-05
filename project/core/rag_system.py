# project/core/rag_system.py

# [Python标准库] 用于生成唯一的会话 ID
import uuid
# [本项目] 导入刚才写好的配置文件
import config

# [第三方库] 导入 LangChain 的 OpenAI 接口类
# 只要是兼容 OpenAI 协议的 API（如硅基流动、DeepSeek官网、Moonshot），都用这个类连接
from langchain_openai import ChatOpenAI

# [本项目] 导入其他核心模块 (保持原样)
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from document_chunker import DocumentChuncker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph


class RAGSystem:

    def __init__(self, collection_name=config.CHILD_COLLECTION):
        # [本项目] 初始化各项资源管理器
        self.collection_name = collection_name
        self.vector_db = VectorDbManager()  # 向量库管理
        self.parent_store = ParentStoreManager()  # 父文档存储管理
        self.chunker = DocumentChuncker()  # 文档切分器

        # 这个变量稍后会存储编译好的 LangGraph 图
        self.agent_graph = None

        # [Python标准库] 生成一个随机的 UUID 作为默认的"线程ID"
        # LangGraph 用这个 ID 来区分不同的用户对话历史
        self.thread_id = str(uuid.uuid4())

    def initialize(self):
        """
        [本项目] 系统初始化核心函数
        负责建立数据库连接、连接 LLM API、并组装 Agent
        """
        # 1. 确保向量数据库集合已创建
        self.vector_db.create_collection(self.collection_name)
        # 获取集合对象，准备传给搜索工具
        collection = self.vector_db.get_collection(self.collection_name)

        # 2. 初始化 LLM (连接硅基流动)
        # [第三方库] 使用 config 中的配置实例化 ChatOpenAI
        llm = ChatOpenAI(
            model=config.LLM_MODEL,  # 模型: deepseek-ai/DeepSeek-V3
            temperature=config.LLM_TEMPERATURE,  # 温度: 0
            openai_api_key=config.SILICONFLOW_API_KEY,  # 从 .env 读到的 Key
            openai_api_base=config.SILICONFLOW_BASE_URL  # 硅基流动的地址
        )

        # 3. 创建工具 (Tools)
        # [本项目] ToolFactory 会把向量库的搜索功能封装成 LLM 可以调用的函数
        # 比如: search_child_chunks(query="...")
        tools = ToolFactory(collection).create_tools()

        # 4. 创建并编译 Agent 图 (Graph)
        # [本项目] 这是最关键的一步！
        # 它把 LLM (大脑) 和 Tools (手) 组装进 graph.py 定义的流程图中
        self.agent_graph = create_agent_graph(llm, tools)

        print(f"✅ 系统初始化完成，已连接模型: {config.LLM_MODEL}")

    def get_config(self):
        """
        [LangGraph] 获取运行配置
        每次调用 graph.invoke 时，都需要传入这个配置，
        LangGraph 会根据里面的 thread_id 找到之前的聊天记录 (Memory)
        """
        return {"configurable": {"thread_id": self.thread_id}}

    def reset_thread(self):
        """
        [本项目] 重置对话
        当用户点击页面上的"清除聊天"按钮时调用
        """
        try:
            # [第三方库] 尝试物理删除当前线程的记忆数据
            if self.agent_graph:
                self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: Could not delete thread {self.thread_id}: {e}")

        # 生成一个新的 ID，对于 LangGraph 来说这就是一个全新的用户
        self.thread_id = str(uuid.uuid4())