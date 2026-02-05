# project/db/parent_store_manager.py

# ============================================================
# 导入部分
# ============================================================

# [Python标准库] json
# 用法：用于读写 JSON 格式的文件。我们将父文档的内容以 JSON 格式存在硬盘上。
import json

# [Python标准库] shutil
# 用法：虽然这里导入了，但本代码中似乎暂时未用到。通常用于文件复制/移动操作。
import shutil

# [本项目] config
# 来源：project/config.py.
# 用法：读取配置中的 PARENT_STORE_PATH (父文档存储文件夹路径)。
import config

# [Python标准库] pathlib.Path
# 用法：现代化的文件路径操作库。
from pathlib import Path

# [Python标准库] typing
# 用法：类型提示，告诉 IDE 参数是字典 (Dict) 还是列表 (List)。
from typing import Dict, List


# ============================================================
# 类定义: ParentStoreManager (父文档存储管理)
# ============================================================
class ParentStoreManager:
    """
    [类功能] 简单的键值对存储 (Key-Value Store)，基于本地文件系统。
    用于存储"父文档"(Parent Chunks) 的完整内容。

    结构：
    - 文件夹: project/parent_store/
    - 文件名: {parent_id}.json
    - 内容: 包含文本内容和元数据的 JSON 对象。
    """

    def __init__(self, store_path=config.PARENT_STORE_PATH):
        # [逻辑] 初始化存储目录
        # 如果目录不存在，自动创建 (parents=True 允许创建多级目录)
        self.__store_path = Path(store_path)
        self.__store_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 保存单个父文档
    # ------------------------------------------------------------
    def save(self, parent_id: str, content: str, metadata: Dict) -> None:
        """
        将一个父文档保存为 JSON 文件。
        """
        # 构造文件路径: store_path/parent_id.json
        file_path = self.__store_path / f"{parent_id}.json"

        # [Python逻辑] 写入文件
        # ensure_ascii=False: 保证中文能正常显示，而不是变成 \uXXXX 乱码
        # indent=2: 格式化 JSON，让人类稍微容易读一点
        file_path.write_text(
            json.dumps({"page_content": content, "metadata": metadata}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    # ------------------------------------------------------------
    # 批量保存
    # ------------------------------------------------------------
    def save_many(self, parents: List) -> None:
        """
        [辅助方法] 循环调用 save 保存多个文档。
        Args:
            parents: 一个包含 (parent_id, document_object) 元组的列表
        """
        for parent_id, doc in parents:
            self.save(parent_id, doc.page_content, doc.metadata)

    # ------------------------------------------------------------
    # 读取单个父文档
    # ------------------------------------------------------------
    def load(self, parent_id: str) -> Dict:
        """
        根据 ID 读取 JSON 文件内容。
        """
        # [逻辑] 兼容性处理
        # 有时候 ID 可能会带有文件后缀，或者大小写不一致，这里尝试做一点容错
        # 优先尝试直接拼接 .json
        file_path = self.__store_path / f"{parent_id}.json"

        # 如果找不到，且 ID 本身已经有后缀了 (虽然在我们的逻辑里 ID 不带后缀)，可以加额外的判断逻辑
        # 这里简化处理：直接读取

        if not file_path.exists():
            return None

        return json.loads(file_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------
    # 公开接口：获取内容
    # ------------------------------------------------------------
    def load_content(self, parent_id: str) -> Dict:
        """
        load 方法的别名，方便外部调用。
        """
        return self.load(parent_id)