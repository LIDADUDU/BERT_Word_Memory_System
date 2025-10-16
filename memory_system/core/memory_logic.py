import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
from memory_system.core.model import SemanticModel


# 确保在文件顶部定义默认常量，但最好在类内部处理路径
DEFAULT_CSV_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "CET6_word_list.csv"
)
DEFAULT_CACHE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "word_vectors_cache.npy"
)


class MemoryLogic:
    """
    负责加载数据、实现字形相似度、融合评分和生成联想推荐。
    """

    def __init__(
        self,
        csv_file: str = DEFAULT_CSV_FILE,
        cache_file: str = DEFAULT_CACHE_FILE,
        alpha: float = 0.6,
        beta: float = 0.4,
        top_n: int = 10,
    ):
        # 将配置参数保存为实例属性
        self.csv_file = csv_file
        self.cache_file = cache_file
        self.ALPHA = alpha
        self.BETA = beta
        self.TOP_N = top_n

        # 1. 加载数据 (使用 self.csv_file)
        self.word_data = self._load_data()
        self.words = self.word_data["word"].tolist()

        # 2. 初始化语义模型 (BERT)

        self.semantic_model = SemanticModel()

        # 3. 缓存所有单词的向量（重要优化！）
        print("正在计算/加载所有单词的 BERT 向量 (利用 GPU)...")
        self.word_vectors = self._get_all_word_vectors()
        print("所有单词向量缓存完成。")

        # 4. 创建字形/同根词索引 (用于快速查找)
        self.rel_words_index = self._create_rel_words_index()
        # 5. 创建同义词索引 (用于 $S_{form}$ 增强)
        self.syn_index = self._create_syn_index()

    def _load_data(self) -> pd.DataFrame:
        """从 CSV 文件加载单词数据，并确保 rel_words 是字符串。"""
        try:
            df = pd.read_csv(self.csv_file)

            # 统一 word 列为小写，确保查找一致性
            df["word"] = df["word"].str.lower()

            # 强制将 rel_words 列转换为字符串
            df["rel_words"] = df["rel_words"].astype(str)
            df["synonyms"] = df["synonyms"].astype(str)

            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"错误：找不到文件 {self.csv_file}，请检查路径。")

    def _get_all_word_vectors(self) -> Dict[str, np.ndarray]:
        """为所有单词生成并缓存向量。"""

        # 尝试加载缓存
        if os.path.exists(self.cache_file):
            print("正在从缓存加载单词向量...")
            try:
                # 使用 np.load 加载保存为 .npy 的字典
                cached_data = np.load(self.cache_file, allow_pickle=True).item()

                # 检查缓存是否与当前词库匹配
                if set(cached_data.keys()) == set(self.words):
                    print("成功加载缓存。")
                    return cached_data
                else:
                    print("警告：词库已更改，重新计算向量。")

            except Exception as e:
                print(f"警告：加载缓存失败 ({e})，将重新计算向量。")

        # 缓存缺失或无效，重新计算
        vectors = {}
        for i, word in enumerate(self.words):
            if i % 100 == 0:
                # 打印进度，避免长时间无响应
                print(f"进度: {i}/{len(self.words)} - 正在计算 '{word}' 的向量...")

            vectors[word] = self.semantic_model.get_word_embedding(word)

        # 保存缓存
        try:
            print(f"计算完成，正在保存向量到缓存文件: {self.cache_file}")
            # 使用 np.save 保存字典
            np.save(self.cache_file, vectors)
        except Exception as e:
            print(f"致命错误：无法保存向量缓存 ({e})。")

        return vectors

    def _create_rel_words_index(self) -> Dict[str, List[str]]:
        """建立一个快速查询索引：哪个单词是哪些其他单词的同根词。"""
        index = {}

        df = self.word_data

        for idx, row in df.iterrows():
            target_word = row["word"]

            rel_words_value = row["rel_words"]

            # 1. 核心防御：如果不是字符串，强制转换为字符串
            if not isinstance(rel_words_value, str):
                rel_words_value = str(rel_words_value)

            # 2. 解析同根词列表。
            rel_list = [
                w.strip()
                for w in rel_words_value.split("|")
                if w.strip() and w.strip().lower() != "nan"
            ]

            for rel_word_raw in rel_list:

                # ***终极防御：在最容易出错的地方，再次强制转换为小写字符串***
                try:
                    # rel_word_raw 理论上是字符串，但我们再次调用 str() 以防万一，并转换为小写
                    rel_word = str(rel_word_raw).lower()
                except Exception as e:
                    # 如果连 str() 转换都失败，说明数据严重异常，跳过。
                    print(
                        f"致命数据警告：无法转换 '{target_word}' 的同根词 '{rel_word_raw}' (类型: {type(rel_word_raw)})，错误: {e}"
                    )
                    continue

                # 记录：rel_word 是 target_word 的同根词
                if rel_word not in index:
                    index[rel_word] = []
                index[rel_word].append(target_word)
        return index

    def _create_syn_index(self) -> Dict[str, List[str]]:
        """建立一个快速查询索引：哪个单词是哪些其他单词的同义词。"""
        index = {}
        df = self.word_data

        for idx, row in df.iterrows():
            target_word = row["word"]

            # 1. 核心防御：确保 syn_words 是字符串
            syn_words_value = row["synonyms"]
            if not isinstance(syn_words_value, str):
                syn_words_value = str(syn_words_value)

            # 2. 解析同义词列表。
            syn_list = [
                w.strip()
                for w in syn_words_value.split("|")
                if w.strip() and w.strip().lower() != "nan"
            ]

            for syn_word_raw in syn_list:
                try:
                    # 终极防御：强制转换为小写字符串
                    syn_word = str(syn_word_raw).lower()
                except Exception:
                    continue

                # 记录：syn_word 是 target_word 的同义词
                if syn_word not in index:
                    index[syn_word] = []
                index[syn_word].append(target_word)
        return index

    # --- S_字形计算 ---
    def calculate_S_form(self, target_word: str, candidate_word: str) -> float:
        """
        计算字形/词汇关系相似度 S_form。
        规则：如果它们互为同根词或互为同义词，则相似。
        """

        target_word_lower = target_word.lower()
        candidate_word_lower = candidate_word.lower()

        # 1. 检查是否为 'rel_words' (同根词) 关系
        is_rel_word = False

        # 1.1. 目标词的 rel_words 列表是否包含候选词
        try:
            target_series = self.word_data.loc[
                self.word_data["word"] == target_word_lower, "rel_words"
            ]

            if not target_series.empty:
                rel_words_str = str(target_series.iloc[0])
                target_rel_words = [
                    w.strip().lower()
                    for w in rel_words_str.split("|")
                    if w.strip() and w.strip().lower() != "nan"
                ]

                if candidate_word_lower in target_rel_words:
                    is_rel_word = True

        except Exception:
            is_rel_word = False

        # 1.2. 候选词是否在目标词的同根词索引中
        if (
            not is_rel_word
            and candidate_word_lower in self.rel_words_index
            and target_word_lower in self.rel_words_index[candidate_word_lower]
        ):
            is_rel_word = True

        if is_rel_word:
            return 1.0  # 确认同根词关系

        # 2. 检查是否为 'synonyms' (同义词) 关系
        is_synonym = False

        # 2.1. 目标词的 synonyms 列表是否包含候选词
        try:
            target_series = self.word_data.loc[
                self.word_data["word"] == target_word_lower, "synonyms"
            ]

            if not target_series.empty:
                syn_words_str = str(target_series.iloc[0])
                target_syn_words = [
                    w.strip().lower()
                    for w in syn_words_str.split("|")
                    if w.strip() and w.strip().lower() != "nan"
                ]

                if candidate_word_lower in target_syn_words:
                    is_synonym = True
        except Exception:
            is_synonym = False

        # 2.2. 候选词是否在目标词的同义词索引中
        if (
            not is_synonym
            and candidate_word_lower in self.syn_index
            and target_word_lower in self.syn_index[candidate_word_lower]
        ):
            is_synonym = True

        if is_synonym:
            return 1.0  # 确认同义词关系

        return 0.0

    def get_association_recommendations(self, target_word: str) -> List[Dict]:
        """
        生成联想记忆推荐列表。
        """
        # 统一使用小写进行查找
        target_word_lower = target_word.lower()
        if target_word_lower not in self.word_vectors:
            return []  # 目标词不在词库中

        target_vec = self.word_vectors[target_word_lower]  # 确保向量也是用小写键查找的
        recommendations = []

        # 遍历词库中的所有单词
        for candidate_word in self.words:
            candidate_word_lower = candidate_word.lower()
            if candidate_word_lower == target_word_lower:
                continue  # 不推荐自己

            # --- 核心修改：使用 Try-Except 捕获任何可能的计算错误 ---
            try:
                candidate_vec = self.word_vectors[candidate_word_lower]

                # 1. 计算语义相似度 S_语义
                S_sem = self.semantic_model.calculate_semantic_similarity(
                    target_vec, candidate_vec
                )

                # 2. 计算字形相似度 S_字形
                S_form = self.calculate_S_form(
                    target_word_lower, candidate_word_lower
                )  # 传入小写

                # 3. 融合计算 S_总
                S_total = self.ALPHA * S_sem + self.BETA * S_form

                # 4. 获取中文含义
                meaning_zh = self.word_data[
                    self.word_data["word"] == candidate_word_lower  # 统一使用小写查找
                ]["meaning_zh"].iloc[0]

                recommendations.append(
                    {
                        "word": candidate_word,
                        "S_total": S_total,
                        "S_semantic": S_sem,
                        "S_form": S_form,
                        "meaning_zh": meaning_zh,
                    }
                )

            except Exception as e:
                # 捕获任何计算或查找错误，并跳过这个候选词
                # print(f"警告: 计算 '{target_word}' 与 '{candidate_word}' 的联想得分时出错: {e}")
                continue

        # 5. 排序并返回 Top N
        recommendations.sort(key=lambda x: x["S_total"], reverse=True)
        return recommendations[: self.TOP_N]
