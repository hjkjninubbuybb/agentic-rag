from langchain_core.messages import HumanMessage

class ChatInterface:
    """
    ChatInterface = 聊天接口层（UI ↔ RAG 系统的中间人）

    它的职责非常单一：
    - 接收“用户输入的文本”
    - 包装成 LangChain 能理解的 Message
    - 调用 Agentic RAG 的 graph
    - 把最终回答取出来，返回给 UI（Gradio / CLI / 其他）
    """

    def __init__(self, rag_system):
        """
        rag_system：整个 RAG 系统的“总控对象”
        通常里面包含：
        - agent_graph（LangGraph 工作流）
        - retriever / db / config
        - 线程 / memory 管理方法
        """
        self.rag_system = rag_system

    def chat(self, message, history):
        """
        核心聊天方法（UI 每发一句话，就会调用一次这个方法）

        参数：
        - message: 当前用户输入的一句话（字符串）
        - history: 历史对话（这里没有直接用，但 UI 框架会传进来）

        返回：
        - 一个字符串（最终要显示给用户的回答）
        """

        # ---------- 1️⃣ 防御性检查 ----------
        # 如果 Agent Graph 还没初始化好，直接返回提示
        if not self.rag_system.agent_graph:
            return "⚠️ System not initialized!"

        try:
            # ---------- 2️⃣ 构造 LangChain 的输入 ----------
            # LangGraph / LangChain 统一用 Message 对象，而不是裸字符串
            # HumanMessage = “这是人说的话”
            result = self.rag_system.agent_graph.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=message.strip()  # 去掉首尾空格
                        )
                    ]
                },
                self.rag_system.get_config()  # graph 运行时配置（memory / thread / callbacks 等）
            )

            # ---------- 3️⃣ 取出最终回答 ----------
            # LangGraph 约定：
            # - messages 是一个“对话列表”
            # - 最后一个 message = 系统最终输出
            return result["messages"][-1].content

        except Exception as e:
            # ---------- 4️⃣ 兜底错误处理 ----------
            # 防止 UI 因为异常直接崩掉
            return f"❌ Error: {str(e)}"

    def clear_session(self):
        """
        清空当前对话会话（memory / thread）

        通常用于：
        - UI 里点“Clear / New Chat”
        - 重置 Agent 的对话上下文
        """
        self.rag_system.reset_thread()
