# project/rag_agent/graph.py

# ============================================================
# 导入部分
# ============================================================

# [第三方库] langgraph.graph
# 来源：LangGraph 框架
# 用法：
# - StateGraph: 用来创建一个"带状态"的流程图。所有的节点共享同一个 State。
# - START: 图的起点（常量）。
# - END: 图的终点（常量）。
from langgraph.graph import START, END, StateGraph

# [第三方库] langgraph.checkpoint.memory
# 来源：LangGraph 框架
# 用法：InMemorySaver 用于把对话历史暂存在内存里。
# 作用：没有它，AI 聊完上一句就忘了下一句。
from langgraph.checkpoint.memory import InMemorySaver

# [第三方库] langgraph.prebuilt
# 来源：LangGraph 预置组件
# 用法：
# - ToolNode: 一个现成的节点，专门用来执行工具函数（比如搜索）。你不用自己写 invoke tool 的逻辑。
# - tools_condition: 一个现成的逻辑判断函数。它会自动检查 "Agent 刚才是不是想调用工具？"。如果是，返回 "tools"；否则返回 END。
from langgraph.prebuilt import ToolNode, tools_condition

# [Python标准库] functools
# 用法：partial(func, arg1=x) 用来固定函数的某些参数。
# 场景：nodes.py 里的函数都需要 llm 参数，但 LangGraph 运行时只传 state。
# 所以我们用 partial 把 llm 提前"绑"在函数上。
from functools import partial

# [本项目] 导入我们自己写的模块
from .graph_state import State, AgentState  # 数据结构
from .nodes import *  # 具体的干活节点
from .edges import *  # 路由逻辑


# ============================================================
# 主函数：创建智能体图
# ============================================================
def create_agent_graph(llm, tools_list):
    """
    [函数功能] 组装所有的积木，返回一个可执行的 Graph 对象
    """

    # [第三方库] llm.bind_tools(tools_list)
    # 用法：告诉大模型"你可以使用这些工具"。
    # 效果：模型不会真的调用工具，而是会生成一个"Tool Call"的 JSON 请求。
    llm_with_tools = llm.bind_tools(tools_list)

    # [第三方库] ToolNode(tools_list)
    # 用法：创建一个节点，它真的会去执行上面的 Tool Call，并返回结果。
    tool_node = ToolNode(tools_list)

    # [第三方库] 初始化记忆检查点
    checkpointer = InMemorySaver()

    print("Compiling agent graph...")

    # ------------------------------------------------------------
    # 第一部分：构建 Agent Subgraph (子图)
    # 作用：这是一个专门负责"查资料"的闭环小流程。
    # ------------------------------------------------------------

    # [第三方库] StateGraph(AgentState)
    # 用法：创建一个新图，指定它使用 AgentState (包含 question, final_answer 等字段)
    agent_builder = StateGraph(AgentState)

    # [第三方库] add_node(name, function)
    # 用法：往图里添加节点。
    # 技巧：这里用 partial 把 llm_with_tools 传给了 agent_node 函数
    agent_builder.add_node("agent", partial(agent_node, llm_with_tools=llm_with_tools))
    agent_builder.add_node("tools", tool_node)
    agent_builder.add_node("extract_answer", extract_final_answer)

    # [第三方库] add_edge(start, end)
    # 用法：定义固定的流向。START -> agent
    agent_builder.add_edge(START, "agent")

    # [第三方库] add_conditional_edges(source, condition_func, mapping)
    # 用法：条件分支。
    # source: 从 "agent" 节点出来后。
    # condition_func: 运行 tools_condition 函数（LangGraph自带）。
    # mapping:
    #   - 如果 tools_condition 返回 "tools" -> 去 "tools" 节点
    #   - 如果 tools_condition 返回 END -> 去 "extract_answer" 节点 (我们自定义的终点)
    agent_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: "extract_answer"}
    )

    # 工具执行完 -> 必须回到 agent 继续思考 (ReAct 循环)
    agent_builder.add_edge("tools", "agent")

    # 提取完答案 -> 结束子图
    agent_builder.add_edge("extract_answer", END)

    # [第三方库] compile()
    # 用法：把图编译成一个可执行对象 (Runnable)。
    agent_subgraph = agent_builder.compile()

    # ------------------------------------------------------------
    # 第二部分：构建 Main Graph (主图)
    # 作用：负责统筹全局 (总结 -> 重写 -> 分发任务 -> 汇总)。
    # ------------------------------------------------------------

    # [第三方库] StateGraph(State)
    # 用法：主图使用全局 State (包含 messages, rewrittenQuestions 等)
    graph_builder = StateGraph(State)

    # 添加节点 (引用 nodes.py 里的函数)
    graph_builder.add_node("summarize", partial(analyze_chat_and_summarize, llm=llm))
    graph_builder.add_node("analyze_rewrite", partial(analyze_and_rewrite_query, llm=llm))
    graph_builder.add_node("human_input", human_input_node)

    # [关键点] 把上面编译好的 agent_subgraph 当作一个节点加入主图！
    # 这就是"图中有图" (Nested Graph)。
    graph_builder.add_node("process_question", agent_subgraph)

    graph_builder.add_node("aggregate", partial(aggregate_responses, llm=llm))

    # 定义流程边
    graph_builder.add_edge(START, "summarize")
    graph_builder.add_edge("summarize", "analyze_rewrite")

    # [本项目] route_after_rewrite
    # 来源：project/rag_agent/edges.py
    # 用法：这是我们自己写的路由逻辑，决定是"人工介入"还是"并行处理"。
    graph_builder.add_conditional_edges("analyze_rewrite", route_after_rewrite)

    # 闭环：人工介入后 -> 回到重写节点
    graph_builder.add_edge("human_input", "analyze_rewrite")

    # 并行处理完 -> 汇总
    graph_builder.add_edge(["process_question"], "aggregate")
    graph_builder.add_edge("aggregate", END)

    # [第三方库] compile(checkpointer=..., interrupt_before=...)
    # 用法：
    # 1. 传入 checkpointer 启用记忆功能。
    # 2. interrupt_before=["human_input"]: 告诉系统，进 human_input 之前暂停！等待用户命令。
    agent_graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_input"]
    )

    print("✓ Agent graph compiled successfully.")
    return agent_graph