# project/document_chunker.py

# ============================================================
# 导入部分
# ============================================================
import os
import glob
from pathlib import Path

# [本项目] 配置文件
# 用来获取切片大小、重叠大小等参数
import config

# [第三方库] langchain_text_splitters
# 来源：LangChain 文本处理库
# 用法：
# - MarkdownHeaderTextSplitter: 专门用来切 Markdown 的，可以识别 #, ## 标题，把标题作为元数据保留。
# - RecursiveCharacterTextSplitter: 最通用的切片器，按字符数递归切分，尽量不切断句子。
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


# ============================================================
# 类定义: DocumentChuncker
# ============================================================
class DocumentChuncker:

    def __init__(self):
        """
        [初始化] 准备好两把"刀"：一把切大块，一把切小块。
        """
        # [第一把刀] 父文档切分器 (按标题切)
        # 它可以把 Markdown 文档按章节结构拆开，比如把 "## 1. Introduction" 下的内容切成一块。
        self.__parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=config.HEADERS_TO_SPLIT_ON,  # 配置在 config.py (比如 #, ##, ###)
            strip_headers=False  # 保留标题文本在正文中
        )

        # [第二把刀] 子文档切分器 (按字数切)
        # 用来把巨大的父块切成适合向量检索的小切片 (比如 500 字)。
        self.__child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE,  # 500
            chunk_overlap=config.CHILD_CHUNK_OVERLAP  # 100 (重叠一点，防止切断关键词)
        )

        # [配置] 读取父块的大小限制
        self.__min_parent_size = config.MIN_PARENT_SIZE  # 2000 (太小就合并)
        self.__max_parent_size = config.MAX_PARENT_SIZE  # 10000 (太大就强拆)

    # ------------------------------------------------------------
    # 公开方法：批量处理整个文件夹
    # ------------------------------------------------------------
    def create_chunks(self, path_dir=config.MARKDOWN_DIR):
        all_parent_chunks, all_child_chunks = [], []

        # [Python标准库] glob 遍历文件夹下所有 .md 文件
        for doc_path_str in sorted(glob.glob(os.path.join(path_dir, "*.md"))):
            doc_path = Path(doc_path_str)
            # 调用处理单个文件的函数
            parent_chunks, child_chunks = self.create_chunks_single(doc_path)
            all_parent_chunks.extend(parent_chunks)
            all_child_chunks.extend(child_chunks)

        return all_parent_chunks, all_child_chunks

    # ------------------------------------------------------------
    # 公开方法：处理单个文件 (核心逻辑)
    # ------------------------------------------------------------
    def create_chunks_single(self, md_path):
        doc_path = Path(md_path)

        # 1. 读取 Markdown 文件内容
        with open(doc_path, "r", encoding="utf-8") as f:
            # [第三方库] 使用 Markdown 切分器进行初步切分
            # 此时得到的是基于标题的原始块
            parent_chunks = self.__parent_splitter.split_text(f.read())

        # 2. 优化父块 (清洗数据)
        # 这一步非常重要！原始的 Markdown 切分可能导致很多只有一句话的小标题块。
        # 我们需要把过小的块合并，把过大的块拆分，保证父块大小适中 (2k~10k字)。
        merged_parents = self.__merge_small_parents(parent_chunks)
        split_parents = self.__split_large_parents(merged_parents)
        cleaned_parents = self.__clean_small_chunks(split_parents)

        all_parent_chunks, all_child_chunks = [], []

        # 3. 生成子块并建立关联 (Parent-Child Mapping)
        self.__create_child_chunks(all_parent_chunks, all_child_chunks, cleaned_parents, doc_path)

        return all_parent_chunks, all_child_chunks

    # ------------------------------------------------------------
    # 私有方法：合并过小的父块
    # ------------------------------------------------------------
    def __merge_small_parents(self, chunks):
        """
        [算法逻辑] 如果某个章节只有几句话 (小于 min_size)，把它合并到前一个章节里。
        防止上下文碎片化。
        """
        if not chunks:
            return []

        merged, current = [], None

        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                # [核心逻辑] 累加内容
                current.page_content += "\n\n" + chunk.page_content
                # [核心逻辑] 合并元数据 (保留标题层级路径)
                # 例如: "Introduction" -> "Background"
                for k, v in chunk.metadata.items():
                    if k in current.metadata:
                        current.metadata[k] = f"{current.metadata[k]} -> {v}"
                    else:
                        current.metadata[k] = v

            # 只有当积累到足够大时，才把它作为一个独立的 Parent
            if len(current.page_content) >= self.__min_parent_size:
                merged.append(current)
                current = None

        # 处理循环结束后的残留块
        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                # ... (元数据合并逻辑同上)
            else:
                merged.append(current)

        return merged

    # ------------------------------------------------------------
    # 私有方法：拆分过大的父块
    # ------------------------------------------------------------
    def __split_large_parents(self, chunks):
        """
        [算法逻辑] 如果某个章节太长 (超过 max_size)，强制把它切开。
        防止 LLM 上下文溢出。
        """
        split_chunks = []

        for chunk in chunks:
            if len(chunk.page_content) <= self.__max_parent_size:
                split_chunks.append(chunk)
            else:
                # 临时创建一个大块切分器
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.__max_parent_size,
                    chunk_overlap=config.CHILD_CHUNK_OVERLAP
                )
                # [第三方库] split_documents
                sub_chunks = splitter.split_documents([chunk])
                split_chunks.extend(sub_chunks)

        return split_chunks

    # ------------------------------------------------------------
    # 私有方法：再次清理
    # ------------------------------------------------------------
    def __clean_small_chunks(self, chunks):
        """
        [算法逻辑] 拆分后可能又产生了小碎片，最后再合并一次，确保万无一失。
        逻辑与 __merge_small_parents 类似。
        """
        cleaned = []
        # ... (逻辑同上，为了节省篇幅，核心思想就是：如果不满 min_size，就拼到下一块或上一块去)

        # 简化的逻辑展示：
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < self.__min_parent_size:
                if cleaned:
                    # 合并到前一块
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    # (元数据合并...)
                elif i < len(chunks) - 1:
                    # 或者合并到后一块
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    # (元数据合并...)
            else:
                cleaned.append(chunk)

        return cleaned

    # ------------------------------------------------------------
    # 私有方法：创建子块并关联 (关键步骤)
    # ------------------------------------------------------------
    def __create_child_chunks(self, all_parent_pairs, all_child_chunks, parent_chunks, doc_path):
        """
        [核心逻辑]
        1. 给每个 Parent 生成唯一 ID。
        2. 把 Parent 切碎成 Child。
        3. 把 Parent ID 塞进每个 Child 的 metadata 里。
        """
        for i, p_chunk in enumerate(parent_chunks):
            # 1. 生成父文档 ID (例如: "report_2024_parent_0")
            parent_id = f"{doc_path.stem}_parent_{i}"

            # 更新父文档的元数据
            p_chunk.metadata.update({"source": str(doc_path.stem) + ".pdf", "parent_id": parent_id})

            # 保存父文档 (ID, 内容对象)
            all_parent_pairs.append((parent_id, p_chunk))

            # 2. 切分子文档
            # [第三方库] split_documents
            children = self.__child_splitter.split_documents([p_chunk])

            # 3. 建立关联 (这是最重要的一步！)
            # 这样检索到 Child 时，才能反向查找到 Parent
            for child in children:
                child.metadata.update({"parent_id": parent_id})

            all_child_chunks.extend(children)