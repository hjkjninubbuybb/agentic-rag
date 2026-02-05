# project/rag_agent/nodes.py

# ============================================================
# 导入部分
# ============================================================

# [第三方库] langchain_core.messages
# 来源：LangChain 框架核心库
# 用法：这些是标准的"消息类"，用来在 Python 代码和大模型之间传递信息。
# - SystemMessage(content="..."):  用来给 AI 设定人设或系统指令。
# - HumanMessage(content="..."):   代表用户说的话。
# - AIMessage(content="..."):      代表 AI 回复的话。
# - RemoveMessage(id="..."):       LangGraph 特有的工具，用于"删除"某条历史消息。
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage

# [本项目] .graph_state
# 来源：我们自己在 project/rag_agent/graph_state.py 定义的
# 用法：导入数据结构类，用于类型提示 (Type Hinting)，让代码知道 state 长什么样。
from .graph_state import State, AgentState

# [本项目] .prompts
# 来源：我们自己在 project/rag_agent/prompts.py 定义的
# 用法：导入函数来获取长长的提示词字符串，避免把这里代码弄乱。
from .prompts import *


# [修改说明] 已注释掉 QueryAnalysis，因为我们现在采用直通模式，不需要结构化输出检查
# from .schemas import QueryAnalysis


# ============================================================
# 节点 1: analyze_chat_and_summarize (对话总结)
# ============================================================
def analyze_chat_and_summarize(state: State, llm):
    """
    [节点功能] 总结历史对话
    """
    # [本项目] state 是一个字典，访问 "messages" 键获取历史记录列表
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}

    # [Python标准库] 列表推导式
    # [第三方库] isinstance(obj, Class)
    # 用法：检查 msg 是否属于 HumanMessage 或 AIMessage 类型。
    # 目的：我们只想总结"人机对话"，不想总结"工具调用结果"（ToolMessage），以免干扰 AI。
    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage))
           and not getattr(msg, "tool_calls", None)  # [第三方库] 检查消息是否包含工具调用请求
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}

    # [Python逻辑] 简单的字符串拼接，把对象列表变成一段可读的文本
    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    # [第三方库] llm.with_config(temperature=...)
    # 来源：LangChain Runnable 协议
    # 用法：创建一个新的 LLM 对象，但修改其配置。
    # 目的：这里临时把 temperature 设为 0.2，让 AI 在总结时稍微灵活一点，而不是死板地复述。
    llm_configured = llm.with_config(temperature=0.2)

    # [本项目] 获取提示词字符串
    prompt_content = get_conversation_summary_prompt()

    # [第三方库] llm.invoke(input)
    # 来源：LangChain 核心方法
    # 用法：这是调用大模型最常用的方法！
    # 参数：input 是一个列表，包含 [系统指令, 用户输入]。
    # 返回：一个 AIMessage 对象，content 属性里就是 AI 生成的文本。
    summary_response = llm_configured.invoke(
        [SystemMessage(content=prompt_content)] +
        [HumanMessage(content=conversation)]
    )

    # [本项目] 返回字典
    # LangGraph 会自动把这个字典合并到全局 State 中。
    # "__reset__": True 是我们在 graph_state.py 里定义的特殊逻辑，用来清空之前的答案。
    return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}


# ============================================================
# 节点 2: analyze_and_rewrite_query (问题处理 - 直通模式)
# ============================================================
def analyze_and_rewrite_query(state: State, llm):
    """
    [节点功能] 强制直通模式 (Skip Query Analysis)
    为了兼容硅基流动/DeepSeek API，跳过结构化检查，直接传递问题。
    """
    # [本项目] 获取用户最新发的一条消息
    last_message = state["messages"][-1]

    # [Python逻辑] 直接提取文本内容，不进行任何 LLM 处理
    # 如果这里调用了 llm.with_structured_output(...)，DeepSeek 可能会因为格式错误报错。
    rewritten_query = last_message.content

    # [Python标准库] 打印调试信息到终端
    print(f"⏩ [直通模式] 跳过澄清检查，直接处理问题: {rewritten_query}")

    # [本项目] 返回更新后的 State
    # 这里必须返回符合 State 定义的字段。
    return {
        "questionIsClear": True,  # 强制设为 True，让 Graph 走向"搜索"分支
        "messages": [],  # 空列表表示不修改历史记录
        "originalQuery": last_message.content,
        "rewrittenQuestions": [rewritten_query]  # 必须是列表 List[str]
    }


# ============================================================
# 节点 3: human_input_node (人工介入)
# ============================================================
def human_input_node(state: State):
    """
    [节点功能] 占位符节点
    """
    # 这个函数什么都不做。
    # 它的作用是作为一个"路标"。在 graph.py 中，我们设置了 interrupt_before=["human_input"]。
    # 所以当流程走到这个节点**之前**，程序会暂停，等待人工操作。
    return {}


# ============================================================
# 节点 4: agent_node (执行搜索 Agent)
# ============================================================
def agent_node(state: AgentState, llm_with_tools):
    """
    [节点功能] ReAct 风格的搜索节点
    """
    # [本项目] 获取提示词
    sys_msg = SystemMessage(content=get_rag_agent_prompt())

    # [本项目] state.get("messages")
    # 检查当前子图（Agent Subgraph）里有没有历史消息。
    if not state.get("messages"):
        # --- 情况 A: 第一次运行 ---
        # 把要解决的问题 (state["question"]) 包装成 HumanMessage
        human_msg = HumanMessage(content=state["question"])

        # [第三方库] llm_with_tools.invoke(...)
        # 这里的 llm_with_tools 是一个"绑定了工具"的模型对象。
        # 用法：和普通的 llm.invoke 一样，但模型现在知道它有权调用 search_child_chunks 等函数。
        # 结果：如果模型决定查资料，response.tool_calls 属性里会有内容。
        response = llm_with_tools.invoke([sys_msg] + [human_msg])

        # 返回更新：把用户的提问和 AI 的回答（或工具调用请求）都存入历史
        return {"messages": [human_msg, response]}

    # --- 情况 B: 后续运行 (工具已经执行完了) ---
    # 这时 state["messages"] 里已经有了：[用户问, AI想查, 工具返回的结果]
    # 我们再次调用 LLM，让它看到工具返回的结果，继续思考。
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# ============================================================
# 节点 5: extract_final_answer (提取答案)
# ============================================================
def extract_final_answer(state: AgentState):
    """
    [节点功能] 从消息历史中提取最终的文本回复
    """
    # [Python标准库] reversed(...) 倒序遍历，从最近的消息开始找
    for msg in reversed(state["messages"]):
        # [第三方库] 检查消息类型
        # 1. 必须是 AI 说的 (AIMessage)
        # 2. 必须有文本内容 (msg.content)
        # 3. 不能是工具调用请求 (not msg.tool_calls) -> 这意味着 AI 认为不需要再查了，给出了最终结论
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            # [本项目] 构造符合 AgentState 定义的返回结构
            res = {
                "final_answer": msg.content,
                "agent_answers": [{
                    "index": state["question_index"],
                    "question": state["question"],
                    "answer": msg.content
                }]
            }
            return res

    # 兜底：如果找了一圈没找到有效回答
    return {
        "final_answer": "Unable to generate an answer.",
        "agent_answers": [{
            "index": state["question_index"],
            "question": state["question"],
            "answer": "Unable to generate an answer."
        }]
    }


# ============================================================
# 节点 6: aggregate_responses (聚合回答)
# ============================================================
def aggregate_responses(state: State, llm):
    """
    [节点功能] 将多个搜索结果汇总成一段话
    """
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    # [Python标准库] sorted(...)
    # 用法：按索引对答案排序，确保回答顺序和问题顺序一致
    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    # [Python标准库] 格式化字符串
    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\nAnswer {i}:\n"f"{ans['answer']}\n")

    # [本项目] 获取提示词
    sys_prompt = get_aggregation_prompt()

    # 构造给 LLM 的最终输入
    user_message = HumanMessage(
        content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")

    # [第三方库] llm.invoke(...)
    # 用法：让 LLM 阅读所有搜索到的片段，写一篇漂亮的总结。
    synthesis_response = llm.invoke(
        [SystemMessage(content=sys_prompt)] +
        [user_message]
    )

    return {"messages": [AIMessage(content=synthesis_response.content)]}