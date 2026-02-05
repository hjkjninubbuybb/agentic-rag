# project/rag_agent/graph_state.py

# [Python标准库] typing 是 Python 用来做"类型提示"的库
# List: 表示一个列表，比如 [1, 2, 3]
# Annotated: 用来给类型加"额外的元数据/功能"，在 LangGraph 中有特殊用途（见下文）
from typing import List, Annotated

# [第三方库] LangGraph 的基础状态类
# MessagesState 内置了一个 'messages' 字段，专门用来存聊天记录（User说啥，AI回啥）
# 我们继承它，就自动拥有了存聊天记录的能力
from langgraph.graph import MessagesState


# [本项目] 自定义的一个"Reducer"函数（归约函数）
# 作用：决定当新数据来的时候，是"追加"到旧列表里，还是"清空"旧列表
# existing: 旧的数据
# new: 新传进来的数据
def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    # [Python基础] 列表推导式 + any()
    # 意思：如果新数据里有任何一个元素包含 key 为 '__reset__' 且值为 True
    if new and any(item.get('__reset__') for item in new):
        # 就清空列表，返回空
        return []
    # 否则，把新数据追加到旧数据后面 (比如 [旧1] + [新1] = [旧1, 新1])
    return existing + new


# [本项目] 定义主图的状态 (Main Graph State)
# 继承自 MessagesState，所以它自动有了 'messages' 字段
class State(MessagesState):
    """State for main agent graph"""

    # [Python基础] 类型注解 (Type Hint)
    # 语法是：变量名: 类型 = 默认值

    # 标记用户的问题是否清晰（True/False）
    questionIsClear: bool = False

    # 存对话的总结（字符串）
    conversation_summary: str = ""

    # 用户最初始的问题（字符串）
    originalQuery: str = ""

    # AI 重写后的问题列表（可能把一个复杂问题拆成了好几个）
    rewrittenQuestions: List[str] = []

    # [LangGraph核心概念] Annotated + Reducer
    # 这里的 agent_answers 存放所有智能体查到的答案。
    # 关键在于 Annotated[List[dict], accumulate_or_reset]：
    # 它的意思是：当有节点往这个字段写数据时，不要直接覆盖！
    # 而是调用我们上面写的 `accumulate_or_reset` 函数，把新数据"追加"进去。
    # 这就是为什么多个 Agent 并行运行时，它们的答案不会互相覆盖，而是汇总到一起。
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []


# [本项目] 定义子图的状态 (Agent Subgraph State)
# 这是给并行运行的小 Agent 用的，它们不需要复杂的 Reducer，只要存自己的临时数据
class AgentState(MessagesState):
    """State for individual agent subgraph"""

    # 当前 Agent 正在处理的那个子问题
    question: str = ""

    # 这个子问题的编号（第几个问题）
    question_index: int = 0

    # 当前 Agent 查到的最终答案文本
    final_answer: str = ""

    # 结构化的答案数据
    agent_answers: List[dict] = []