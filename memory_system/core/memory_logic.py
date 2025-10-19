import pandas as pd
import numpy as np
import os
from typing import List, Dict

# 假设 SemanticModel 可以在同级目录或父级路径中导入
from .semantic_model import SemanticModel
import Levenshtein
from tqdm import tqdm

# 导入 Jaro-Winkler 相似度函数（假设您已安装 python-Levenshtein 库）
from Levenshtein import jaro_winkler
import math


# --- 文件路径定义 ---
# 基于当前文件 (memory_logic.py) 的相对路径定义
# 🌟 修复/确认：路径使用新的 CET6_word_list.csv
DEFAULT_CSV_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "CET6_word_list.csv"
)
DEFAULT_CACHE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "word_vectors_cache.npy"
)
# 语义相似度矩阵缓存文件路径
DEFAULT_SIMILARITY_MATRIX_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "semantic_similarity_matrix.npy"
)
# --- 文件路径定义 ---


class MemoryLogic:
    """
    负责加载数据、实现字形相似度、融合评分和生成联想推荐。
    已优化：引入语义相似度矩阵预计算，提高联想推荐的实时响应速度。
    """

    def __init__(
        self,
        csv_file: str = DEFAULT_CSV_FILE,
        cache_file: str = DEFAULT_CACHE_FILE,
        matrix_file: str = DEFAULT_SIMILARITY_MATRIX_FILE,  # 新增参数
        alpha: float = 0.6,
        beta: float = 0.4,
        top_n: int = 10,
    ):
        # 将配置参数保存为实例属性
        self.csv_file = csv_file
        self.cache_file = cache_file
        self.matrix_file = matrix_file  # 保存矩阵文件路径
        self.ALPHA = alpha
        self.BETA = beta
        self.TOP_N = top_n

        # 1. 加载数据
        self.word_data = self._load_data()
        self.words = self.word_data["word"].tolist()
        # 创建 word 到其在 self.words 列表中索引的映射
        self.word_to_index = {word: i for i, word in enumerate(self.words)}
        self.word_data.set_index(
            "word", inplace=True
        )  # 优化：设置 word 为索引，方便快速查找含义

        # 2. 初始化语义模型 (BERT)
        # 🌟 假设 SemanticModel 能够被正确导入，否则这里会失败
        self.semantic_model = SemanticModel()

        # 3. 缓存所有单词的向量
        print("正在计算/加载所有单词的 BERT 向量 (利用 GPU)...")
        self.word_vectors = self._get_all_word_vectors()
        print("所有单词向量缓存完成。")

        # 4. 🌟 新增：计算/加载语义相似度矩阵
        print("正在计算/加载语义相似度矩阵 (用于快速查询)...")
        self.semantic_matrix = self._get_all_semantic_similarity()
        print("语义相似度矩阵加载完成。")

        # 5. 创建字形/同根词索引
        self.rel_words_index = self._create_rel_words_index()

    def _load_data(self) -> pd.DataFrame:
        """从 CSV 文件加载单词数据，并进行基本预处理。"""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"未找到词库文件: {self.csv_file}")

        # keep_default_na=False 确保空值（如 rel_words 字段）被视为 '' 而不是 NaN
        word_data = pd.read_csv(self.csv_file, keep_default_na=False)
        word_data["word"] = word_data["word"].str.lower()

        # 🌟 修复：确保所有必需的列存在或被填充，以防音标和词性查找失败
        for col in ["pos", "meaning_zh", "phonetic_us", "phonetic_uk"]:
            if col not in word_data.columns:
                word_data[col] = ""  # 假设缺失时为空字符串
            else:
                word_data[col] = word_data[col].fillna("")

        return word_data

    def _get_all_word_vectors(self) -> Dict[str, np.ndarray]:
        """加载或计算所有单词的 BERT 向量并缓存。"""
        # ... (此方法逻辑保持不变)
        if os.path.exists(self.cache_file):
            print("正在从缓存加载单词向量...")
            try:
                # 必须使用 allow_pickle=True 来加载 Dict 对象
                cache = np.load(self.cache_file, allow_pickle=True).item()
                if set(cache.keys()) == set(self.words):
                    print("成功加载缓存。")
                    return cache
                else:
                    print("⚠️ 缓存与当前词库不匹配，将重新计算。")
            except Exception:
                print("⚠️ 缓存文件加载失败，将重新计算。")
        else:
            print("未找到缓存文件，将计算所有单词向量。")

        # 重新计算逻辑
        word_vectors = {}

        # 🌟 核心修改 2: 使用 tqdm 包装 self.words 列表，显示计算进度
        for word in tqdm(self.words, desc="计算 BERT 向量"):
            try:
                word_vectors[word] = self.semantic_model.get_word_embedding(word)
            except Exception:
                print(f"❌ 警告：无法获取词汇 '{word}' 的向量，跳过。")
                continue

        # 保存到缓存
        try:
            np.save(self.cache_file, word_vectors)
        except Exception:
            pass  # 忽略保存失败

        return word_vectors

    def _get_all_semantic_similarity(self) -> np.ndarray:
        """
        优化核心：计算或加载所有单词的语义相似度矩阵 (N x N)。
        """
        # ... (此方法逻辑保持不变)
        N = len(self.words)

        # 1. 尝试从缓存加载矩阵
        if os.path.exists(self.matrix_file):
            print("正在从缓存加载语义相似度矩阵...")
            try:
                matrix = np.load(self.matrix_file)
                if matrix.shape == (N, N):
                    print("成功加载相似度矩阵。")
                    return matrix
                else:
                    print(
                        f"⚠️ 缓存矩阵形状不匹配 ({matrix.shape} vs {(N, N)})，将重新计算。"
                    )
            except Exception as e:
                print(f"⚠️ 缓存矩阵加载失败 ({e})，将重新计算。")
        else:
            print("未找到相似度矩阵缓存，开始计算。")

        # 2. 重新计算矩阵
        matrix = np.zeros((N, N), dtype=np.float32)

        # 提取所有向量并堆叠成一个 NumPy 数组 (提高效率)
        # 修复：确保 only words with vectors are included
        words_with_vectors = [word for word in self.words if word in self.word_vectors]
        N_valid = len(words_with_vectors)

        if N_valid != N:
            print(f"❌ 警告：{N - N_valid} 个词汇向量缺失，将使用部分矩阵。")

        # 优化：只需计算上三角矩阵
        total_steps = N * (N - 1) // 2
        pbar = tqdm(total=total_steps, desc="计算相似度矩阵")

        for i in range(N):
            vec_i = self.word_vectors.get(self.words[i])
            if vec_i is None:
                # 补偿跳过的进度步数：由于 i 的循环已经完成，我们跳过 j > i 的部分
                pbar.update(N - 1 - i)
                continue

            # 从 i 开始，计算与自己的相似度 (1.0) 和与其他词的相似度
            for j in range(i, N):
                vec_j = self.word_vectors.get(self.words[j])
                if vec_j is None:
                    if j > i:  # 只有当 j > i 时，才需要更新 pbar
                        pbar.update(1)
                    continue

                # 🌟 注意：这里假设 semantic_model.calculate_semantic_similarity 接受 NumPy 数组
                sim = self.semantic_model.calculate_semantic_similarity(vec_i, vec_j)
                matrix[i, j] = sim
                matrix[j, i] = sim  # 对称填充

                if j > i:
                    pbar.update(1)  # 只在计算非对角线元素时更新进度条

        pbar.close()

        # 3. 保存到缓存
        try:
            np.save(self.matrix_file, matrix)
        except Exception:
            print("❌ 警告：相似度矩阵保存失败。")

        return matrix

    def _create_rel_words_index(self) -> Dict[str, str]:
        # ... (此方法逻辑保持不变)
        """创建所有同根词(rel_words)到主词(word)的映射索引。"""
        rel_words_index = {}
        for main_word, row in self.word_data.iterrows():  # 🌟 优化：使用 iterrows 遍历
            rel_words_str = str(row["rel_words"])

            rel_words_index[main_word] = main_word

            if rel_words_str:
                for rel_word in rel_words_str.split("|"):
                    rel_word_stripped = rel_word.strip().lower()
                    if rel_word_stripped and rel_word_stripped not in rel_words_index:
                        rel_words_index[rel_word_stripped] = main_word

        return rel_words_index

    def calculate_S_form(self, word1: str, word2: str) -> float:
        # ... (此方法逻辑保持不变)
        """
        优化：计算两个单词的字形相似度 S_Form (0.0 到 1.0)。
        融合了同根词检查、Jaro-Winkler 和 Levenshtein 相似度，以提高准确性。
        """

        # 1. 检查是否为同根词 (来自 CSV 预设)
        root1 = self.rel_words_index.get(word1)
        root2 = self.rel_words_index.get(word2)

        if root1 and root1 == root2:
            return 1.0  # 预设的同根词给予满分

        # 2. 如果不是同根词，计算基于字符串的相似度
        max_len = max(len(word1), len(word2))

        if max_len == 0:
            return 0.0

        # Levenshtein 相似度
        distance = Levenshtein.distance(word1, word2)
        sim_lev = 1.0 - (distance / max_len)

        # Jaro-Winkler 相似度 (通常在名字和单词匹配上表现更好)
        sim_jaro_winkler = jaro_winkler(word1, word2)

        # 融合两种相似度，并给予更高的权重给 Jaro-Winkler
        similarity = 0.6 * sim_jaro_winkler + 0.4 * sim_lev

        return max(0.0, similarity)

    def _get_full_display_word(self, word: str, word_info: pd.Series) -> str:
        """从 Series 中获取音标，组合成 'word [phonetic]' 形式。"""
        # 尝试获取音标
        phonetic = ""
        try:
            # 优先使用美式音标
            if word_info.get("phonetic_us"):
                phonetic = word_info["phonetic_us"]
            # 其次使用英式音标
            elif word_info.get("phonetic_uk"):
                phonetic = word_info["phonetic_uk"]
        except KeyError:
            # 如果音标列不存在，则跳过
            pass

        # 组合单词和音标 (例如: "word [phonetic]")
        if phonetic:
            # 去除音标字符串可能存在的额外括号，然后用方括号包围
            phonetic_clean = phonetic.strip().strip("[]")
            return f"{word} [{phonetic_clean}]"
        else:
            return word

    def get_word_meaning_by_word(self, word: str) -> str:
        """从加载的数据中快速查找单词的中文含义，并预置词性。"""
        try:
            # 1. 查找对应的行数据
            word_info = self.word_data.loc[word.lower()]
            # 2. 提取词性和含义
            pos = word_info["pos"]
            meaning_zh = word_info["meaning_zh"]
            # 3. 组合并返回 (e.g., "n. 名词含义")
            return f"{pos}. {meaning_zh}"
        except KeyError:
            return "（含义缺失）"

    # 🌟 新增方法：获取完整的显示单词（含音标）
    def get_full_display_word(self, word: str) -> str:
        """获取带音标的完整显示单词字符串。"""
        try:
            word_info = self.word_data.loc[word.lower()]
            return self._get_full_display_word(word.lower(), word_info)
        except KeyError:
            return word  # 找不到则返回原始单词

    def get_association_recommendations(
        self,
        target_word: str,
        mode: str = "total",  # 接受新的模式参数
        threshold: float = 0.4,  # 调整默认阈值：防止过滤掉所有结果
        top_n: int | None = None,  # 修复：添加 top_n 参数，允许运行时覆盖 self.TOP_N
    ) -> List[Dict]:
        """
        优化核心：使用预计算的语义相似度矩阵，将 O(N) 实时计算降为 O(1) 查找。
        为目标词计算并返回联想词推荐列表，支持按模式和阈值过滤。
        """

        # 确定最终要返回的数量，如果 top_n 未提供，则使用实例属性
        final_top_n = top_n if top_n is not None else self.TOP_N

        target_word_lower = target_word.lower()
        if target_word_lower not in self.word_to_index:
            return []

        # 1. 确定目标词的索引
        target_index = self.word_to_index[target_word_lower]

        # 2. 从预计算矩阵中取出目标词的语义相似度（O(1) 操作）
        # S_sem_vector 是一个包含所有词汇语义相似度分数的 N 维向量
        S_sem_vector = self.semantic_matrix[target_index, :]

        recommendations = []

        # 3. 遍历所有候选词 (O(N) 循环，但移除了耗时的 BERT 计算)
        for i, candidate_word in enumerate(self.words):
            candidate_word_lower = candidate_word

            if target_index == i:
                continue

            try:
                # 实时查询 S_语义：直接从矩阵中获取 (O(1))
                S_sem = S_sem_vector[i]

                # 实时计算 S_字形：仍然需要实时计算，但比 BERT 快得多
                S_form = self.calculate_S_form(target_word_lower, candidate_word_lower)

                # 4. 根据 mode 确定最终评分和过滤值
                score = 0.0
                should_keep = True

                S_total_actual = self.ALPHA * S_sem + self.BETA * S_form

                if mode == "total":
                    score = S_total_actual
                elif mode == "semantic":
                    score = S_sem
                elif mode == "form":
                    score = S_form
                else:
                    should_keep = False  # 遇到未知模式，跳过

                # 应用阈值过滤
                if score < threshold:
                    should_keep = False

                if not should_keep:
                    continue

                # 5. 记录结果
                recommendations.append(
                    {
                        "word": candidate_word,
                        # 🌟 修复: 强制转换为 Python 原生 float，确保 JSON 序列化成功
                        "S_total": float(S_total_actual),
                        "S_semantic": float(S_sem),
                        "S_form": float(S_form),
                        "meaning_zh": self.get_word_meaning_by_word(candidate_word),
                        "primary_score": float(score),  # 此次排序/过滤的依据
                    }
                )

            except Exception as e:
                # print(f"错误处理词汇 {candidate_word}: {e}")
                continue

        # 6. 排序：按 primary_score 降序排序
        recommendations.sort(key=lambda x: x["primary_score"], reverse=True)

        # 7. 返回结果：返回 final_top_n
        return recommendations[:final_top_n]
