# 📖 基于 BERT 的单词联想记忆系统

## 🌟 项目简介

本项目旨在通过结合**语义相似度**和**字形相似度**（同根词/同义词）的融合评分模型，为学习者提供高效、多维度的单词联想记忆推荐。

**核心模型：**
$$S_{Total} = \alpha \times S_{Semantic} + \beta \times S_{Form}$$

- **$S_{Semantic}$ (语义相似度):** 基于 BERT 模型生成的词嵌入向量的余弦相似度。
- **$S_{Form}$ (字形相似度):** 基于词根/衍生词列表的二元匹配（匹配则为 1.0，否则为 0.0）。

## 📁 项目结构

```

[PROJECT\_ROOT]/
├── memory\_system/
│   ├── core/
│   │   ├── memory\_logic.py        \# 核心业务逻辑 (MemoryLogic 类)
│   │   ├── model.py               \# BERT 语义模型封装 (SemanticModel 类)
│   │   └── **init**.py
│   ├── data/
│   │   ├── CET6\_word\_list.csv     \# 词库数据 (输入)
│   │   └── word\_vectors\_cache.npy \# BERT 向量缓存 (自动生成)
│   ├── main.py                    \# 命令行入口 (argparse 解析)
│   └── **init**.py                \# 声明 Python 顶级包
├── venv\_memory\_system/
├── requirements.txt               \# 项目依赖列表 (精简版)
└── README.md

```

## 🛠️ 环境配置与安装

### 1. 环境要求

- Python 3.8+
- GPU 支持 (推荐使用 NVIDIA GPU 加速 BERT 向量计算，非必须)

### 2. 依赖安装

请确保您已在项目根目录激活虚拟环境（如 `venv_memory_system`），然后安装依赖：

```bash
# 1. 切换到项目根目录
cd [YOUR_PROJECT_PATH]

# 2. 激活虚拟环境 (示例：根据您的环境路径进行调整)
.\venv_memory_system\Scripts\activate

# 3. 安装所有依赖
pip install -r requirements.txt
```

_(注：第一次运行时，BERT 模型将自动下载。)_

## 🚀 使用指南 (命令行接口 CLI)

**⚠️ 关键：** 请务必在项目根目录使用 **`python -m` 模块执行模式** 启动 `memory_system.main`。直接执行 `python memory_system/main.py` 将导致导入错误。

### 1\. 查询特定单词

在命令行中输入要查询的单词：

```bash
# 查询单词 'trade' 的联想词 (使用默认配置)
python -m memory_system.main trade
```

### 2\. 自定义评分权重和推荐数量 (高级)

用户可以通过命令行参数自定义评分权重 ($\alpha, \beta$) 和推荐数量 (`top-n`)。

| 参数           | 描述                         | 默认值 |
| :------------- | :--------------------------- | :----- |
| `[query_word]` | 要查询的单词（位置参数）     | (None) |
| `--alpha`      | $S_{Semantic}$ (语义) 的权重 | `0.6`  |
| `--beta`       | $S_{Form}$ (字形) 的权重     | `0.4`  |
| `--top-n`      | 返回的联想词数量             | `10`   |

**示例 1：更看重字形联想 (增加 $\beta$ 的权重，并只看 Top 5)**

```bash
python -m memory_system.main credit --alpha 0.4 --beta 0.6 --top-n 5
```

**示例 2：运行默认测试案例 (未提供单词，但使用自定义权重)**

```bash
python -m memory_system.main --alpha 0.8 --beta 0.2
```

## 📝 贡献与致谢

- **数据来源：** https://github.com/kajweb/dict?tab=readme-ov-file
- **模型技术：** Hugging Face `transformers` 库，BERT 预训练模型
- **作者/贡献者:** [李大嘟嘟]
- **GitHub：** https://github.com/LIDADUDU
