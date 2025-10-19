import pandas as pd
import os
import json
import logging
import math
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class AssociationLogic:
    """
    轻量级的联想逻辑类，仅处理词汇数据加载和联想推荐计算。
    它使用预计算的联想矩阵，不涉及 SM2 算法或数据库操作。
    """

    # 基于当前文件的相对路径来定位数据
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    # 路径优化：确保跨平台兼容
    CSV_FILE_PATH = os.path.join(
        BASE_DIR, "memory_system", "data", "CET6_word_list.csv"
    )
    ASSOCIATION_FILE_PATH = os.path.join(
        BASE_DIR, "memory_system", "data", "association_matrix.json"
    )

    def __init__(self, alpha: float = 0.6, beta: float = 0.4, top_n: int = 10):
        # 权重和 Top-N
        self.ALPHA = alpha
        self.BETA = beta
        self.TOP_N = top_n

        # 1. 加载静态词汇数据
        self.word_data, self.word_map = self._load_word_data()
        self.words = list(self.word_map.keys())

        # 2. 加载联想矩阵
        self._association_matrix = self._load_association_matrix()

        logger.info(f"AssociationLogic 初始化完成，静态词汇数: {len(self.word_map)}")

    def _load_word_data(self) -> tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
        """从 CSV 文件加载词汇数据，返回 DataFrame 和用于快速查找的 map。"""
        if not os.path.exists(self.CSV_FILE_PATH):
            logger.error(f"❌ 词库文件未找到: {self.CSV_FILE_PATH}")
            return pd.DataFrame(), {}
        try:
            # 读取 CSV
            df = pd.read_csv(self.CSV_FILE_PATH, encoding="utf-8")
            df["word"] = df["word"].astype(str).str.strip().str.lower()
            df.set_index("word", inplace=True)

            # word_map 字典用于快速查询中文含义
            word_map = df.to_dict(orient="index")
            return df, word_map
        except Exception as e:
            logger.error(f"❌ 加载词库文件 {self.CSV_FILE_PATH} 失败: {e}")
            return pd.DataFrame(), {}

    def _load_association_matrix(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """从 JSON 文件加载预计算的联想权重数据，若失败则回退到模拟。"""
        if os.path.exists(self.ASSOCIATION_FILE_PATH):
            try:
                with open(self.ASSOCIATION_FILE_PATH, "r", encoding="utf-8") as f:
                    matrix = json.load(f)
                    logger.info("成功加载预计算的联想权重矩阵。")
                    return matrix
            except Exception as e:
                logger.error(f"加载联想权重文件失败: {e}。使用模拟数据。")

        # 回退到模拟数据
        return self._simulate_association_matrix_fallback()

    def _simulate_association_matrix_fallback(
        self,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """原始的硬编码模拟数据，作为回退方案。"""
        matrix = {}
        # 仅为 'credit' 和 'create' 创建模拟联想
        sim_data = {
            "create": {
                "credit": {"S_semantic": 0.3, "S_form": 0.8},
                "credible": {"S_semantic": 0.25, "S_form": 0.75},
                "creed": {"S_semantic": 0.1, "S_form": 0.6},
            },
            "credit": {
                "create": {"S_semantic": 0.3, "S_form": 0.8},
                "credible": {"S_semantic": 0.9, "S_form": 0.85},
                "creed": {"S_semantic": 0.6, "S_form": 0.7},
                "decade": {"S_semantic": 0.1, "S_form": 0.1},
            },
        }

        for word, sim_words in sim_data.items():
            matrix[word] = {}
            for candidate, scores in sim_words.items():
                if candidate in self.word_map:
                    matrix[word][candidate] = scores
        return matrix

    def get_association_recommendations(
        self, target_word: str, mode: str = "total", threshold: float = 0.7
    ) -> Dict[str, List[Dict]]:
        """
        为目标词计算并返回联想词推荐列表，使用预计算的联想矩阵。
        """
        target_word_lower = target_word.lower()
        recommendations = []

        if target_word_lower not in self._association_matrix:
            # 如果没有预计算数据，则尝试返回前缀匹配作为弱联想 (仅用于回退)
            potential_matches = [
                word
                for word in self.words
                if word.startswith(target_word_lower) and word != target_word_lower
            ]

            for word in potential_matches:
                s_form_sim = (
                    len(target_word_lower) / len(word) if len(word) > 0 else 0.0
                )
                s_total = (
                    self.ALPHA * 0.1 + self.BETA * s_form_sim
                )  # 默认给一个低的语义分

                meaning_zh = self.word_map.get(word, {}).get(
                    "meaning_zh", "（含义缺失）"
                )

                recommendations.append(
                    {
                        "word": word,
                        "S_total": round(s_total, 4),
                        "S_semantic": 0.1,
                        "S_form": round(s_form_sim, 4),
                        "meaning_zh": meaning_zh,
                        "primary_score": s_total,
                    }
                )
        else:
            # 从预计算矩阵中获取数据
            for candidate_word, scores in self._association_matrix.get(
                target_word_lower, {}
            ).items():

                S_sem = scores.get("S_semantic", 0.0)
                S_form = scores.get("S_form", 0.0)
                S_total = self.ALPHA * S_sem + self.BETA * S_form
                primary_score = 0.0

                if mode == "total":
                    primary_score = S_total
                elif mode == "semantic":
                    primary_score = S_sem
                elif mode == "form":
                    primary_score = S_form
                else:
                    continue

                # 阈值过滤
                if primary_score < threshold:
                    continue

                meaning_zh = self.word_map.get(candidate_word, {}).get(
                    "meaning_zh", "（含义缺失）"
                )

                recommendations.append(
                    {
                        "word": candidate_word,
                        "S_total": round(S_total, 4),
                        "S_semantic": round(S_sem, 4),
                        "S_form": round(S_form, 4),
                        "meaning_zh": meaning_zh,
                        "primary_score": primary_score,
                    }
                )

        # 排序：按 primary_score 降序排序
        recommendations.sort(key=lambda x: x["primary_score"], reverse=True)

        return {"recommendations": recommendations[: self.TOP_N]}
