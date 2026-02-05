# project/rag_agent/edges.py

# ============================================================
# 导入部分
# ============================================================

# [Python标准库] typing
# Literal: 限制变量只能是特定的几个值
# List: 列表类型
from typing import Literal, List

# [第三方库] langgraph.types.Send
# 来源：LangGraph 框架
# 用法：用于 Map-Reduce 模式，把任务分发给多个子图并发执行
from langgraph.types import Send

# [本项目] .graph_state
# 作用：引入 State 类用于类型检查
from .graph_state import State


# ============================================================
# 路由逻辑
# ============================================================

# [修改点] 更新了返回值的类型注解 (Type Hint)
# 意思是：这个函数要么返回字符串 "human_input"，要么返回一个 Send 对象的列表
def route_after_rewrite(state: State) -> Literal["human_input"] | List[Send]:
    """
    [函数功能] 决定流程的下一步走向
    被 graph.py 中的 add_conditional_edges 调用。
    """

    # [逻辑] 检查问题是否清晰
    # 如果我们在 nodes.py 里用了"直通模式"，这里的 questionIsClear 通常都是 True
    if not state.get("questionIsClear", False):
        # 情况 A: 问题不清 -> 返回节点名称 (字符串)
        # 流程将走向 "human_input" 节点
        return "human_input"

    else:
        # [核心逻辑] 并行分发 (Map Step)
        # 情况 B: 问题清晰 -> 返回 Send 对象列表
        # LangGraph 会根据这个列表，启动 N 个并行的 "process_question" 子图

        # 假设 state["rewrittenQuestions"] 是 ["问题1", "问题2"]
        # 这里就会生成两个 Send 对象
        return [
            # [第三方库] Send(node_name, state)
            # node: 目标节点名 ("process_question")
            # arg: 传给该子图的初始状态
            Send(
                "process_question",
                {
                    "question": query,  # 分配给这个 Agent 的具体问题
                    "question_index": idx,  # 问题的编号 (方便最后排序汇总)
                    "messages": []  # 初始化该子图的消息历史为空
                }
            )
            for idx, query in enumerate(state["rewrittenQuestions"])
        ]