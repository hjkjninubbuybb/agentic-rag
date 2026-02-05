# project/rag_agent/schemas.py

# ============================================================
# 导入部分
# ============================================================
from typing import List
from pydantic import BaseModel, Field


# ============================================================
# 类定义: QueryAnalysis (查询分析模型 - 中文版)
# ============================================================
class QueryAnalysis(BaseModel):
    """
    [类功能] 定义 LLM 输出的"格式契约"。
    """

    # [字段 1] 问题是否清晰
    is_clear: bool = Field(
        description="判断用户的问题是否清晰、明确且可以根据文档进行回答。"
    )

    # [字段 2] 重写后的查询列表
    questions: List[str] = Field(
        description="一个包含重写后查询语句的列表。将用户模糊的问题转化为独立的、适合检索的具体问题。"
    )

    # [字段 3] 需要澄清的内容
    clarification_needed: str = Field(
        description="如果 is_clear 为 False，请在此详细解释为什么问题不清晰，以及需要用户提供什么补充信息。"
    )