import csv
import random
import os
import sys
from datetime import datetime, timedelta
from flask_cors import CORS
from flask import Flask, jsonify, request, render_template

# 从 Python 标准库导入 ConnectionError
import socket  # 用于网络错误捕获

# --- 引入 MongoDB 驱动 ---
try:
    from pymongo import MongoClient

    # 引入 PyMongoError 作为所有 pymongo 错误的基类
    from pymongo.errors import PyMongoError
except ImportError as e:
    print("❌ 错误：导入 pymongo 失败。请检查依赖！")
    print(f"详细错误: {e}")
    sys.exit(1)


# 引入 SBERT 优化的核心逻辑
# 🌟 依赖于执行时项目的根目录在 Python PATH 中
try:
    # 确保这里的路径正确
    from memory_system.core.memory_logic import MemoryLogic
except ImportError:
    # 捕获导入错误后，打印详细信息
    print(
        "❌ 致命错误：无法导入 MemoryLogic。请检查路径：memory_system/core/memory_logic.py"
    )
    print("请确认您的项目结构：app.py 和 memory_system 文件夹在同一级。")
    print(f"当前 Python 查找路径: {sys.path}")
    sys.exit(1)


app = Flask(__name__)
CORS(app)
app.config["JSON_AS_ASCII"] = False

# ----------------------------
# 1. 配置与初始化核心服务
# ----------------------------

# ❗ 关键配置：请替换为您的 MongoDB 连接字符串 ❗
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "vocabulary_system"
COLLECTION_NAME = "sm2_states"

DEMO_USER_ID = "guest_user_123"

# 全局数据库和集合对象
mongo_client = None
sm2_collection = None
MEMORY_LOGIC_SERVICE = None
WORD_DF = None  # 用于存储词汇基础数据的 DataFrame


def initialize_db_connection():
    """初始化 MongoDB 连接并检查数据库状态。"""
    global mongo_client, sm2_collection

    print(f"➡️ 正在尝试连接 MongoDB: {MONGO_URI} ...")
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # 尝试执行一个操作来验证连接
        mongo_client.admin.command("ping")

        db = mongo_client[DATABASE_NAME]
        sm2_collection = db[COLLECTION_NAME]

        print("✅ MongoDB 连接成功。")
    # 捕获 PyMongoError 和标准网络错误
    except (PyMongoError, socket.error) as e:
        print(
            f"❌ 致命错误：MongoDB 连接失败。请检查 MongoDB 服务器是否运行和 URI 是否正确。\n错误详情: {e}"
        )
        mongo_client = None
        sm2_collection = None


def initialize_system_services():
    """初始化 MemoryLogic (BERT/SBERT) 和数据持久化。"""
    global MEMORY_LOGIC_SERVICE, WORD_DF

    # --- 1. 初始化 SBERT 联想服务 ---
    try:
        print("✨ 正在初始化 MemoryLogic 服务 (高性能 BERT 架构)...")
        # 🌟 MemoryLogic 初始化将触发数据加载和矩阵计算
        MEMORY_LOGIC_SERVICE = MemoryLogic()
        WORD_DF = (
            MEMORY_LOGIC_SERVICE.word_data.reset_index()
        )  # 恢复 DataFrame 索引，以防需要
        print("✅ MemoryLogic 初始化完成。")
    except Exception as e:
        print(f"❌ 致命错误：MemoryLogic 初始化失败。错误：{e}")
        return False

    # --- 2. 初始化 MongoDB 连接 ---
    initialize_db_connection()
    if sm2_collection is None:
        print("⚠️ 警告：数据库连接失败，应用将无法持久化 SM2 状态。")
        return True

    # --- 3. 确保所有词汇的基础 SM2 状态存在于 MongoDB 中 ---
    if WORD_DF is None or WORD_DF.empty:
        print("❌ 警告：无法获取词汇数据，跳过 SM2 状态初始化。")
        return True

    print("➡️ 正在检查和初始化 MongoDB 中的 SM2 状态...")
    today_str = datetime.now().strftime("%Y-%m-%d")

    # 批量检查和插入缺失的记录
    new_records = []

    # 🌟 修复 KeyError: 'raw_word' 🌟
    # 获取用户已有的词汇状态，同时查询 'raw_word' 和 'word' 字段用于兼容旧数据。
    existing_raw_words = set()

    # 查找时同时查询 raw_word 和 word，以处理旧数据结构
    for doc in sm2_collection.find(
        {"user_id": DEMO_USER_ID}, {"raw_word": 1, "word": 1}
    ):
        raw_word = doc.get("raw_word")

        # 如果 raw_word 缺失（旧数据），则从 word 字段中提取原始词汇
        if raw_word is None and "word" in doc:
            # 假设旧数据的 word 字段可能包含 "apple [ˈæpl]" 或 "apple"
            raw_word = doc["word"].split(" ", 1)[0].lower()

        if raw_word:
            existing_raw_words.add(raw_word)

    # 从 MemoryLogic 中获取原始单词列表
    all_raw_words = MEMORY_LOGIC_SERVICE.words

    for raw_word in all_raw_words:
        if raw_word not in existing_raw_words:
            try:
                # 1. 获取带音标的完整显示单词 (e.g., "apple [ˈæpl]")
                display_word = MEMORY_LOGIC_SERVICE.get_full_display_word(raw_word)

                # 2. 获取带词性的含义 (e.g., "n. 苹果")
                full_meaning_zh = MEMORY_LOGIC_SERVICE.get_word_meaning_by_word(
                    raw_word
                )

            except KeyError:
                print(
                    f"❌ 警告：无法在 WORD_DF 中通过索引 '{raw_word}' 找到完整信息，跳过该词初始化。"
                )
                continue

            # 初始化新的 SM2 状态
            record = {
                "user_id": DEMO_USER_ID,
                "word": display_word,  # 存储带音标的显示词汇
                "raw_word": raw_word,  # 存储原始词汇，用于 SM2 查找
                "meaning_zh": full_meaning_zh,  # 存储带词性的含义
                "easiness_factor": 2.5,
                "interval": 0,
                "repetitions": 0,
                "next_review_date": today_str,
                "created_at": datetime.now(),
            }
            new_records.append(record)

    if new_records:
        sm2_collection.insert_many(new_records)
        print(
            f"✅ 成功为用户 '{DEMO_USER_ID}' 插入了 {len(new_records)} 条新的 SM2 记录。"
        )
    else:
        print("✅ 所有词汇状态均已存在。")

    return True


# ----------------------------
# 2. SM-2 算法实现
# ----------------------------
def sm2_update_logic(word_data, quality):
    """
    SM-2 算法计算新的状态值。
    """
    ef = word_data["easiness_factor"]
    n = word_data["repetitions"]
    i = word_data["interval"]

    if quality < 3:
        n = 0
        i = 1
    else:
        ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        if ef < 1.3:
            ef = 1.3

        n += 1

        if n == 1:
            i = 1
        elif n == 2:
            i = 6
        else:
            i = round(i * ef)

    if i < 1:
        i = 1

    next_review_date = datetime.now() + timedelta(days=i)

    # 返回更新后的数据字典
    updated_data = {
        "easiness_factor": ef,
        "repetitions": n,
        "interval": i,
        "next_review_date": next_review_date.strftime("%Y-%m-%d"),
        "updated_at": datetime.now(),
    }
    return updated_data


# ----------------------------
# 3. Web 路由 (必须添加的首页路由)
# ----------------------------


@app.route("/")
def index():
    """渲染主页模板，作为应用入口。"""
    # 确保 templates 文件夹在 app.py 的同级目录下
    return render_template("index.html")


# ----------------------------
# 4. API 路由实现 (使用 MongoDB)
# ----------------------------


def check_service_status():
    """检查核心服务的初始化状态。"""
    if MEMORY_LOGIC_SERVICE is None:
        return (
            jsonify(
                {"status": "error", "error": "Backend logic failed to initialize."}
            ),
            500,
        )
    return None


@app.route("/api/review_list", methods=["GET"])
def review_list():
    """
    GET: 获取当前用户待复习的单词列表（从 MongoDB）。

    返回的 'word' 字段是带音标的；'meaning_zh' 字段是带词性的。
    """
    error_response = check_service_status()
    if error_response:
        return error_response

    if sm2_collection is None:
        return (
            jsonify({"status": "error", "error": "Database service is not available."}),
            503,
        )

    today = datetime.now().strftime("%Y-%m-%d")

    try:
        query = {"user_id": DEMO_USER_ID, "next_review_date": {"$lte": today}}

        # 返回 'word' (带音标) 和 'meaning_zh' (带词性)
        review_words_cursor = sm2_collection.find(
            query, {"word": 1, "meaning_zh": 1, "_id": 0}
        )

        review_words = list(review_words_cursor)

        random.shuffle(review_words)

        return jsonify(
            {"status": "success", "count": len(review_words), "words": review_words}
        )

    except Exception as e:
        return (
            jsonify({"status": "error", "error": f"Database query failed: {str(e)}"}),
            500,
        )


@app.route("/api/update_memory", methods=["POST"])
def update_memory():
    """
    POST: 根据用户评分 (quality) 更新单词的 SM-2 记忆状态（到 MongoDB）。

    关键：SM2 更新使用原始单词 (raw_word) 进行查找。
    """
    error_response = check_service_status()
    if error_response:
        return error_response

    if sm2_collection is None:
        return (
            jsonify({"status": "error", "error": "Database service is not available."}),
            503,
        )

    try:
        data = request.get_json()
        word = data.get("word")  # 接收的可能是带音标的 'display_word'

        if not word or data.get("quality") is None:
            return (
                jsonify({"status": "error", "error": "Invalid input parameters."}),
                400,
            )

        # 1. 提取原始单词：如果输入是 "apple [ˈæpl]", 提取 "apple"
        raw_word_match = word.split(" ", 1)[0].lower()
        quality = data.get("quality")

        if quality < 0 or quality > 5:
            return (
                jsonify({"status": "error", "error": "Invalid quality parameter."}),
                400,
            )

        # 2. 从 MongoDB 读取当前状态。优先使用 'raw_word' 查找，但如果旧数据没有 'raw_word'
        # 我们需要先尝试用 raw_word_match 找到记录。

        # 在新版本中，我们总是使用 raw_word 查找，这确保了准确性
        current_state = sm2_collection.find_one(
            {"user_id": DEMO_USER_ID, "raw_word": raw_word_match}
        )

        # 兼容旧数据（如果旧数据中没有 raw_word，只用 word 存储）
        if not current_state:
            current_state = sm2_collection.find_one(
                {"user_id": DEMO_USER_ID, "word": raw_word_match}
            )

        if not current_state:
            return (
                jsonify(
                    {
                        "status": "error",
                        "error": f"Raw word '{raw_word_match}' record not found for user.",
                    }
                ),
                404,
            )

        # 3. 计算新的 SM-2 状态
        updated_sm2_fields = sm2_update_logic(current_state, quality)

        # 4. 更新 MongoDB 中的记录。我们使用 _id 更新以确保准确性
        result = sm2_collection.update_one(
            {"_id": current_state["_id"]}, {"$set": updated_sm2_fields}
        )

        if result.matched_count == 0:
            return (
                jsonify(
                    {"status": "error", "error": "Failed to match record for update."}
                ),
                500,
            )

        return jsonify(
            {
                "status": "success",
                "message": f"Memory updated for {raw_word_match}.",
                "details": {
                    "next_review_date": updated_sm2_fields["next_review_date"],
                    "new_interval": updated_sm2_fields["interval"],
                    "new_ef": round(updated_sm2_fields["easiness_factor"], 2),
                },
            }
        )

    except Exception as e:
        print(f"Update memory error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/recommend", methods=["GET"])
def recommend():
    """
    GET: 返回联想词推荐列表。

    修正：推荐结果中的 'word' 字段包含音标。
    """
    error_response = check_service_status()
    if error_response:
        return error_response

    word = request.args.get("word", "").strip().lower()  # 原始单词
    mode = request.args.get("mode", "total")
    threshold = float(request.args.get("threshold", 0.4))
    top_n = int(request.args.get("top_n", 10))

    if not word:
        return jsonify({"status": "error", "error": "Word parameter is required."}), 400

    try:
        # 1. 获取基于原始单词的推荐列表
        recommendations = MEMORY_LOGIC_SERVICE.get_association_recommendations(
            target_word=word, mode=mode, threshold=threshold, top_n=top_n
        )

        # 2. 修正推荐列表中的 'word' 字段，使其包含音标
        # 注意：这里我们遍历结果并使用 MemoryLogic 的方法获取带音标的完整显示词汇
        final_recommendations = []
        for rec in recommendations:
            raw_word = rec["word"]
            rec["word"] = MEMORY_LOGIC_SERVICE.get_full_display_word(raw_word)
            final_recommendations.append(rec)

        return jsonify(
            {
                "status": "success",
                "query_word": word,
                "recommendations": final_recommendations,
            }
        )
    except Exception as e:
        print(f"联想推荐错误: {e}")
        return (
            jsonify(
                {"status": "error", "error": f"Error during recommendation: {str(e)}"}
            ),
            500,
        )


@app.route("/api/add_word", methods=["POST"])
def add_word_to_review():
    """
    POST: 将新词汇添加到 MongoDB 中的学习列表。
    """
    error_response = check_service_status()
    if error_response:
        return error_response
    if sm2_collection is None:
        return (
            jsonify({"status": "error", "error": "Database service is not available."}),
            503,
        )

    try:
        data = request.get_json()
        raw_word = data.get("word", "").strip().lower()  # 用户输入的原始词汇

        if not raw_word:
            return (
                jsonify({"status": "error", "error": "Word parameter is required."}),
                400,
            )

        # 1. 检查词汇是否存在于基础数据中，并获取完整信息
        try:
            # 1.1 获取带音标的完整显示单词
            display_word = MEMORY_LOGIC_SERVICE.get_full_display_word(raw_word)
            # 1.2 获取带词性的含义
            full_meaning_zh = MEMORY_LOGIC_SERVICE.get_word_meaning_by_word(raw_word)

        except KeyError:
            return (
                jsonify(
                    {
                        "status": "error",
                        "error": f"Word '{raw_word}' not found in vocabulary base data (Index Lookup Failed).",
                    }
                ),
                404,
            )

        # 2. 检查用户是否已存在该词汇的学习记录 (使用原始词汇raw_word查找)
        existing_record = sm2_collection.find_one(
            {"user_id": DEMO_USER_ID, "raw_word": raw_word}  # 查找时使用 raw_word
        )

        if existing_record:
            return (
                jsonify(
                    {
                        "status": "error",
                        "error": f"Word '{raw_word}' already exists in your learning list.",
                    }
                ),
                409,
            )

        # 3. 创建初始 SM2 记录并插入 MongoDB
        record = {
            "user_id": DEMO_USER_ID,
            "word": display_word,  # 存储带音标的显示词汇
            "raw_word": raw_word,  # 存储原始词汇，用于 SM2 查找
            "meaning_zh": full_meaning_zh,  # 存储带词性的含义
            "easiness_factor": 2.5,
            "interval": 0,
            "repetitions": 0,
            "next_review_date": datetime.now().strftime("%Y-%m-%d"),  # 立即复习
            "created_at": datetime.now(),
        }

        sm2_collection.insert_one(record)

        return jsonify(
            {
                "status": "success",
                "message": f"Word '{raw_word}' added to the learning list.",
                "details": {
                    "word": display_word,
                    "meaning": full_meaning_zh,
                },  # 返回带音标的显示词汇
            }
        )

    except Exception as e:
        print(f"Add word error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


# ----------------------------
# 4. 运行应用
# ----------------------------

if __name__ == "__main__":
    # 强制所有 SM2 状态记录使用带音标的新格式，需要清空旧数据
    print(
        "📢 提示：由于数据结构（word/raw_word）发生变化，建议清空旧的 MongoDB 记录（sm2_states 集合），以便新的初始化逻辑生效。"
    )
    if initialize_system_services():
        app.run(debug=True, port=5000)
    else:
        print("❌ 应用程序初始化失败，请检查控制台输出的致命错误信息。")
