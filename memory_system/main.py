# main.py
import argparse
from typing import List, Dict
from memory_system.core.memory_logic import MemoryLogic

# ----------------------------------------------------------------------
# 打印功能
# ----------------------------------------------------------------------


def print_recommendations(
    target_word: str,
    recommendations: List[Dict],
    alpha: float,
    beta: float,
    top_n: int,
):
    """格式化并打印推荐结果"""
    if not recommendations:
        print(f"⚠️ 注意：未找到 '{target_word}' 的联想结果。")
        return

    print(f"\n查询词: '{target_word}'")

    print(f"Top {top_n} 联想词 (S_总 = {alpha} * S_语义 + {beta} * S_字形):")

    for rec in recommendations:
        # 只打印小数点后4位和1位
        total_score = f"{rec['S_total']:.4f}"
        sem_score = f"{rec['S_semantic']:.4f}"
        form_score = f"{rec['S_form']:.1f}"

        # 使用 f-string 对齐输出
        print(
            f"  - {rec['word']:<15} (S_总: {total_score} | S_sem: {sem_score} | S_form: {form_score}) - {rec['meaning_zh']}"
        )


# ----------------------------------------------------------------------
# 命令行入口
# ----------------------------------------------------------------------


def run_cli():
    """通过命令行界面运行联想推荐系统，并支持自定义参数"""

    # 1. 定义和解析命令行参数
    parser = argparse.ArgumentParser(description="单词联想记忆系统 CLI")

    # query_word 现在是一个可选的位置参数
    parser.add_argument(
        "query_word",
        nargs="?",
        default=None,
        help="要查询的单词（可选，如果未提供则运行默认测试案例）。",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.6, help="S_语义的权重 (默认: 0.6)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.4, help="S_字形的权重 (默认: 0.4)"
    )
    parser.add_argument(
        "--top-n", type=int, default=10, help="推荐联想词的数量 (默认: 10)"
    )

    args = parser.parse_args()

    print("--- 步骤 III: 业务逻辑与融合测试 ---")

    # 2. 实例化逻辑层，传入解析后的参数
    # 如果用户提供了参数，logic将使用这些参数。如果没有，则使用 default=0.6/0.4/10
    logic = MemoryLogic(alpha=args.alpha, beta=args.beta, top_n=args.top_n)

    # 从 logic 实例中获取实际使用的配置，以防将来 logic 内部有校验或修改
    alpha, beta, top_n = logic.ALPHA, logic.BETA, logic.TOP_N

    if args.query_word:
        # 用户提供了查询词
        query_word = args.query_word.lower()
        recs = logic.get_association_recommendations(query_word)
        # 传入实际使用的配置参数
        print_recommendations(query_word, recs, alpha, beta, top_n)
    else:
        # 用户未提供查询词，运行默认测试案例 (现在默认测试也会使用自定义的 alpha/beta/top-n)
        print("\n--- 运行默认测试案例 (使用当前配置) ---")

        # 默认测试 1: 语义强关联
        test_word_sem = "trade"
        recs_sem = logic.get_association_recommendations(test_word_sem)
        print_recommendations(test_word_sem, recs_sem, alpha, beta, top_n)

        # 默认测试 2: 字形强关联
        test_word_form = "action"
        recs_form = logic.get_association_recommendations(test_word_form)
        print_recommendations(test_word_form, recs_form, alpha, beta, top_n)


if __name__ == "__main__":
    # 我们应该在 main.py 中打印 GPU 状态
    try:
        from memory_system.core.model import (
            SemanticModel,
        )  # 假设 SemanticModel 在 main.py 所在目录也可见

        SemanticModel.print_gpu_status()  # 假设你有一个静态方法来打印这个
    except Exception:
        pass  # 如果导入失败或没有这个方法，跳过

    run_cli()
