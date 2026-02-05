# Agentic RAG 项目架构说明书

本文档用于辅助理解本项目的文件结构、类关系及核心运行逻辑。

---

## 1. 文件作用清单 (File Manifest)

### 根目录与配置
* **`project/app.py`**: **【入口】** 程序的启动文件。负责创建 Gradio UI 并启动本地服务器。
* **`project/config.py`**: **【配置】** 全局配置中心。存放 API Key、模型名称、数据库路径、切片参数等常量。
* **`requirements.txt`**: **【依赖】** Python 依赖库列表。

### 核心系统 (Core System)
* **`project/core/rag_system.py`**: **【心脏】** `RAGSystem` 类。系统的总容器，负责初始化数据库、连接 LLM、编译智能体图 (Graph)。
* **`project/core/document_manager.py`**: **【文档管家】** `DocumentManager` 类。负责文档的上传、转换 (PDF->MD)、切片、以及存入向量库和文件存储。
* **`project/core/chat_interface.py`**: **【中介】** `ChatInterface` 类。连接前端 UI 与后端 Agent Graph，处理对话请求和异常。

### 数据存储 (Database Layer)
* **`project/db/vector_db_manager.py`**: **【向量库】** `VectorDbManager` 类。管理 Qdrant 客户端，负责 Embedding 模型加载、集合创建及向量搜索。
* **`project/db/parent_store_manager.py`**: **【父文档库】** `ParentStoreManager` 类。管理本地 JSON 文件存储，用于存取大段的父文档内容。

### 文档处理 (Processing)
* **`project/document_chunker.py`**: **【切片器】** `DocumentChuncker` 类。实现**父子索引 (Parent-Child)** 策略：先按标题切父块，再按字符切子块。
* **`project/util.py`**: **【工具】** PDF 转 Markdown 的辅助函数。

### 智能体大脑 (Agent / LangGraph)
* **`project/rag_agent/graph.py`**: **【总指挥】** 定义 LangGraph 的图结构（节点与边的连接方式），组装 LLM 和工具。
* **`project/rag_agent/nodes.py`**: **【执行官】** 定义具体的节点逻辑函数（如：总结对话、重写问题、执行搜索、聚合答案）。
* **`project/rag_agent/edges.py`**: **【路由】** 定义条件边逻辑（如：判断是否需要人工介入、并发执行搜索）。
* **`project/rag_agent/tools.py`**: **【工具箱】** `ToolFactory` 类。将数据库查询能力封装为 LLM 可调用的 Tool。
* **`project/rag_agent/prompts.py`**: **【剧本】** 存放所有节点的 System Prompt。
* **`project/rag_agent/graph_state.py`**: **【记忆】** 定义 `State` 数据结构，用于在节点间传递数据。

### 用户界面 (UI)
* **`project/ui/gradio_app.py`**: **【前端】** 定义 Gradio 界面布局（Tab、按钮、聊天框）。
* **`project/ui/css.py`**: **【样式】** 自定义 CSS 样式。

---

## 2. 类与模块调用关系 (Hierarchy)

### 初始化层级 (Initialization)
当 `app.py` 启动时，`RAGSystem` 被实例化，它持有以下核心组件：

```text
RAGSystem (单例核心)
├── vector_db      -> VectorDbManager (Qdrant)
├── parent_store   -> ParentStoreManager (JSON Files)
├── chunker        -> DocumentChuncker (Splitter)
└── agent_graph    -> Compiled LangGraph (智能体工作流)