import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 全局配置 ---
# 自动选择 CUDA (GPU) 或 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"  # 使用未区分大小写的通用基础 BERT 模型

# 检查是否成功识别到 GPU
if DEVICE.type == "cuda":
    print(f"✅ 成功检测到 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ 未检测到 CUDA，将使用 CPU 运行模型。")


class SemanticModel:
    """
    负责 BERT 模型的加载、向量化和语义相似度计算。
    """

    def __init__(self):
        # 实例化分词器和模型
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        self.model = BertModel.from_pretrained(MODEL_NAME).to(DEVICE)
        self.model.eval()  # 设置为评估模式，关闭 dropout

    def get_word_embedding(self, word: str) -> np.ndarray:
        """
        将单个单词转换为其 BERT 词嵌入向量。
        我们使用 [CLS] 标记的输出作为整个单词的句子嵌入。

        Args:
            word: 要向量化的单词字符串。

        Returns:
            np.ndarray: 768 维的词嵌入向量。
        """
        # 使用 [CLS] token 获取句子级嵌入。
        # 对于单个单词，[CLS] token 也是一个很好的语义表示。

        # 1. 分词和转换为 Tensor
        inputs = self.tokenizer(
            word, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(
            DEVICE
        )  # 将输入数据移动到 GPU

        # 2. 通过模型获取输出
        with torch.no_grad():  # 推理时关闭梯度计算，节省内存和时间
            outputs = self.model(**inputs)

        # 3. 提取 [CLS] token 的输出 (对应 index 0)
        # last_hidden_state 的形状是 (batch_size, sequence_length, hidden_size)
        # 我们取 batch 0, sequence index 0 (即 [CLS]) 的向量
        cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()

        return cls_embedding

    def calculate_semantic_similarity(
        self, vec1: np.ndarray, vec2: np.ndarray
    ) -> float:
        """
        计算两个词嵌入向量之间的余弦相似度。

        Args:
            vec1, vec2: 768 维的词嵌入向量。

        Returns:
            float: 相似度得分 (范围通常在 -1.0 到 1.0)。
        """
        # reshape(1, -1) 将向量转换为二维数组，满足 sklearn 的输入要求
        return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]


# --- 测试示例 ---
if __name__ == "__main__":

    # 首次运行时会下载 BERT 模型
    print("正在加载 BERT 模型...")
    sm = SemanticModel()
    print("模型加载完成。")

    # 定义测试词汇
    word_a = "trade"
    word_b = "commerce"  # 语义相近的词 (CSV中的同义词)
    word_c = "television"  # 语义不相近的词

    # 1. 向量化
    print(f"\n正在向量化词汇...")
    vec_a = sm.get_word_embedding(word_a)
    vec_b = sm.get_word_embedding(word_b)
    vec_c = sm.get_word_embedding(word_c)
    print(f"{word_a} 向量维度: {vec_a.shape}")  # 预期 (768,)

    # 2. 计算语义相似度
    sim_ab = sm.calculate_semantic_similarity(vec_a, vec_b)
    sim_ac = sm.calculate_semantic_similarity(vec_a, vec_c)

    print("\n--- 语义相似度测试结果 ---")
    print(
        f"'{word_a}' (贸易) vs '{word_b}' (商业) 相似度 (S_语义): {sim_ab:.4f} (预期高分)"
    )
    print(
        f"'{word_a}' (贸易) vs '{word_c}' (电视) 相似度 (S_语义): {sim_ac:.4f} (预期低分)"
    )

    # 预期结果：sim_ab 应该远高于 sim_ac
