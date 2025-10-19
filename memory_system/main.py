import argparse
from typing import List, Dict
import logging

# 导入新的轻量级联想逻辑类
from memory_system.core.association_logic import AssociationLogic

# ----------------------------------------------------------------------
# 🌟 日志配置 🌟
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 打印功能
# ----------------------------------------------------------------------


def print_recommendations(
    target_word: str,
    recommendations: List[Dict],
    mode: str,
    threshold: float,
    alpha: float,
    beta: float,
    top_n: int,
):
    """格式化并打印推荐结果"""
    if not recommendations:
        print(f"⚠️ 注意：未找到 '{target_word}' 的联想结果。")
        return

    print(f"\n查询词: '{target_word}'")

    if mode == "total":
        print(
            f"Top {len(recommendations)} 联想词 (S_总 = {alpha} * S_语义 + {beta} * S_字形):"
        )
    else:
        score_name = "S_语义" if mode == "semantic" else "S_字形"
        print(
            f"联想模式: {score_name} 相似度。仅输出相似度 >= {threshold:.2f} 的词汇 ({len(recommendations)} 个):"
        )

    for rec in recommendations:
        # 统一打印 4 位小数
        total_score = f"{rec['S_total']:.4f}"
        sem_score = f"{rec['S_semantic']:.4f}"
        form_score = f"{rec['S_form']:.4f}"

        # 打印 primary_score，即本次排序/过滤的依据
        primary_score = f"{rec['primary_score']:.4f}"

        # 使用 f-string 对齐输出
        print(
            f"  - {rec['word']:<15} (主要得分: {primary_score} | S_sem: {sem_score} | S_form: {form_score}) - {rec['meaning_zh']}"
        )


# ----------------------------------------------------------------------
# 命令行入口
# ----------------------------------------------------------------------


def run_cli():
    """通过命令行界面运行联想推荐系统，并支持自定义参数"""

    # 1. 定义和解析命令行参数
    parser = argparse.ArgumentParser(description="单词联想记忆系统 CLI")

    parser.add_argument(
        "query_word",
        nargs="?",
        default=None,
        help="要查询的单词（可选，如果未提供则进入调优模式）。",
    )
    # 模式和阈值
    parser.add_argument(
        "--mode",
        type=str,
        default="total",
        choices=["total", "semantic", "form"],
        help="联想模式：total (融合), semantic (词义), form (词形)。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="当 mode 不为 total 时，用于过滤联想词的相似度阈值 (默认: 0.8)。",
    )
    # 权重和 Top-N (仅 mode='total' 时生效)
    parser.add_argument(
        "--alpha", type=float, default=0.7, help="S_语义的权重 (默认: 0.7)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.3, help="S_字形的权重 (默认: 0.3)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="推荐联想词的数量 (默认: 50)。",
    )
    # 调优词
    parser.add_argument(
        "--tuning-word",
        type=str,
        default="credit",
        help="在未提供 query_word 时，指定进行权重调优的单词 (默认: credit)。",
    )

    args = parser.parse_args()

    print("--- 步骤 III: 业务逻辑与融合测试 (CLI) ---")

    # 2. 实例化逻辑层
    # 使用 AssociationLogic
    logic = AssociationLogic(alpha=args.alpha, beta=args.beta, top_n=args.top_n)

    # 从 logic 实例中获取实际使用的配置
    alpha, beta, top_n = logic.ALPHA, logic.BETA, logic.TOP_N

    if args.query_word:
        # 用户提供了查询词 (标准运行模式)
        query_word = args.query_word.lower()

        # *** 关键调用：接收字典并解包 ***
        recs_dict = logic.get_association_recommendations(
            query_word, mode=args.mode, threshold=args.threshold
        )
        recs = recs_dict.get("recommendations", [])

        # 传入所有参数给打印函数
        print_recommendations(
            query_word, recs, args.mode, args.threshold, alpha, beta, top_n
        )
    else:
        # 用户未提供查询词：进入调优模式
        tuning_word = args.tuning_word.lower()
        run_tuning(logic, tuning_word)


def run_tuning(logic: AssociationLogic, target_word: str):
    """
    运行超参数调优测试，遍历不同的 alpha/beta 组合，并显示结果对比。
    """
    print(f"\n--- 🌟 权重调优模式 (Tuning Mode) 🌟 ---")
    print(f"目标词: '{target_word}'")
    print(f"注意: 此模式仅测试 'total' 模式下的权重组合。")

    # 调优参数范围：从 0.1 到 0.9，步长 0.1
    alpha_values = [round(i * 0.1, 1) for i in range(1, 10)]

    for alpha in alpha_values:
        beta = round(1.0 - alpha, 1)

        # 临时更新 logic 实例的权重进行测试
        logic.ALPHA = alpha
        logic.BETA = beta
        # 强制设置 top_n 为 3，以便观察对比
        logic.TOP_N = 3

        # 重新计算联想词
        recs_dict = logic.get_association_recommendations(
            target_word, mode="total", threshold=0.0
        )
        recs = recs_dict.get("recommendations", [])

        # 打印本次配置的结果摘要 (只看 Top 3)
        if recs:
            summary = []
            for rec in recs:
                # 仅显示单词名和总分
                summary.append(f"{rec['word']} ({rec['S_total']:.4f})")

            print(f"| α={alpha:.1f}, β={beta:.1f} | Top 3: {' -> '.join(summary)}")
        else:
            print(f"| α={alpha:.1f}, β={beta:.1f} | Top 3: (未找到联想词)")

    # 恢复默认 TOP_N
    logic.TOP_N = 50


if __name__ == "__main__":
    run_cli()
