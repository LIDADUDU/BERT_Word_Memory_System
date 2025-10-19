# 📖 基于 BERT 和融合评分模型的单词联想记忆系统

## 🌟 项目简介 (Introduction)

本项目是一个功能完备的 Web 应用，旨在通过结合**高性能自然语言处理（NLP）技术**和**认知科学**原理，为学习者提供下一代单词记忆体验。

系统核心在于：

1.  **联想模型：** 结合 **BERT 词嵌入**与**字形匹配**的**融合评分模型**，提供多维度的单词联想推荐。
2.  **复习策略：** 采用经典的 **SM-2 间隔重复算法**，实现科学、智能的复习调度。
3.  **技术栈：** 后端基于 **Flask** 框架，使用 **MongoDB** 进行灵活的数据持久化，前端提供直观的 Web UI。

## 🔬 核心模型与算法 (Core Model & Algorithm)

### 1. 联想评分融合模型

联想推荐的评分 $S_{Total}$ 结合了语义相似度 $S_{Semantic}$ 和字形相似度 $S_{Form}$：

$$S_{Total} = \alpha \times S_{Semantic} + \beta \times S_{Form}$$

| 评分项             | 描述                           | 核心技术                                                                                                                     | 优化                                                           |
| :----------------- | :----------------------------- | :--------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------- |
| **$S_{Semantic}$** | 单词在上下文中的意义相似度。   | 基于 **Hugging Face `transformers` 库**，使用 **`bert-base-uncased`** 预训练模型提取词嵌入向量（`[CLS]` 标记）的余弦相似度。 | 预计算相似度矩阵并缓存（`.npy`），支持 **GPU/CUDA** 加速计算。 |
| **$S_{Form}$**     | 单词的词根、词缀或拼写相似度。 | 基于字形距离算法（如 Jaro-Winkler）或词根匹配。                                                                              |
| **SM-2 算法**      | 决定单词下次复习的日期和间隔。 | 在 `memory_logic.py` 中实现，根据用户对单词卡片的 0-5 评分实时更新 **EF (熟练度因子)** 和 **I (间隔天数)**。                 |

---

## 📁 项目结构 (File Structure)

```

基于NLP技术的单词记忆系统/
├── .git/
├── memory\_system/                  \# 核心应用代码和库
│   ├── core/                       \# 业务逻辑核心模块
│   │   ├── db\_instance.py          \# 数据库抽象层（配置 MongoDB 连接）
│   │   ├── memory\_logic.py         \# SM-2 算法和融合评分逻辑
│   │   ├── semantic\_model.py       \# BERT 模型加载与向量化
│   ├── data/                       \# 原始数据和模型缓存
│   │   └── semantic\_similarity\_matrix.npy \# 预计算的 BERT 相似度矩阵
│   ├── association\_logic.py        \# 联想推荐 API 逻辑
│   └── main.py                     \# 命令行接口 (CLI) 入口
├── templates/                      \# Flask 网页模板
│   └── index.html                  \# Web UI 主界面
├── app.py                          \# Web 应用 (Flask) 主入口，负责路由和初始化服务
├── requirements.txt                \# 项目依赖包列表
└── README.md                       \# 本文件

```

## ⚙️ 安装与运行 (Installation & Usage)

### 1. 环境要求 (Prerequisites)

- Python 3.8+
- **MongoDB 服务器**：必须运行（本地或远程）。
- GPU 支持 (推荐使用，非必须)。

### 2. 依赖安装

```bash
# 1. 切换到项目根目录
cd [YOUR_PROJECT_PATH]

# 2. 激活虚拟环境 (示例)
source .venv/bin/activate

# 3. 安装所有依赖
pip install -r requirements.txt
```

### 3\. 数据准备

1.  **MongoDB URI 配置：** 确保 `app.py` 或配置文件中的 `MONGO_URI` 正确指向您的 MongoDB 实例。
2.  **模型初始化：** 首次运行时，BERT 模型将自动下载 (`semantic_model.py`)，并且项目需要时间计算或加载预计算的语义相似度矩阵。

### 4\. 运行 Web 应用 (推荐方式)

应用启动后，它将同时初始化 MongoDB 连接、加载 BERT 模型，并开始监听 HTTP 请求。

```bash
# 确保 MongoDB 服务已启动
# 启动 Flask Web 应用：
python app.py
```

应用启动后，请在浏览器中访问：

```
[http://127.0.0.1:5000/](http://127.0.0.1:5000/)
```

> **提示：** 首页路由 `@app.route("/")` 在 `app.py` 中被定义，用于渲染 `index.html`。

## 👨‍💻 核心模块详细说明 (Core Modules Overview)

| 文件                    | 核心技术                                        | 主要功能                                                                                                               |
| :---------------------- | :---------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| **`app.py`**            | Flask, PyMongo                                  | **Web 应用主入口。** 定义所有 API 路由，包括 `/` 首页、`/next_review_word` 和 `/rate_word` 等，并初始化 MongoDB 连接。 |
| **`memory_logic.py`**   | SM-2, BERT                                      | **核心调度和评分。** 实现了 SM-2 算法，负责融合评分（$S_{Total}$）的计算和复习日期的管理。                             |
| **`semantic_model.py`** | **Hugging Face `transformers` (BERT)**, PyTorch | **NLP 模型接口。** 负责加载 **`bert-base-uncased`** 模型，提供词嵌入向量和语义相似度计算，支持 GPU 加速。              |
| **`db_instance.py`**    | MongoDB (PyMongo)                               | **数据库抽象。** 封装 MongoDB 的连接和客户端实例，作为数据持久化的统一入口。                                           |
| **`index.html`**        | HTML, JS, Tailwind CSS                          | **前端 UI。** 实现了单词卡片展示、SM-2 评分交互、联想提示和 Tab 切换等功能。                                           |

---

### 许可证 (License)

本项目采用 [LICENSE 文件中指定的许可证]。

```

```

## 📝 贡献与致谢

- **数据来源：** https://github.com/kajweb/dict?tab=readme-ov-file
- **许可证：** MIT License
- **作者/贡献者:** [李大嘟嘟]
- **GitHub：** https://github.com/LIDADUDU
