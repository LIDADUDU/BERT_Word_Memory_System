import json
import csv
import os

# --- 配置 ---
INPUT_FILE = "CET6_2.json"
OUTPUT_FILE = "CET6_word_list.csv"

# CSV 文件中我们需要的字段（表头）
FIELDNAMES = [
    "word",
    "meaning_zh",
    "pos",
    "phonetic_uk",
    "phonetic_us",
    "synonyms",
    "rel_words",
    "exam_level",
]


def extract_word_data(word_entry):
    """
    从单个单词的 JSON 字典中提取所需数据。
    V2 修复：确保所有关键路径都能被正确访问。
    """

    # 1. 有效性检查：必须是字典，且必须有 headWord 和 content 路径
    if not isinstance(word_entry, dict) or not word_entry.get("headWord"):
        return None

    # 获取核心数据结构，如果没有，则条目无效
    word_data = word_entry.get("content", {}).get("word", {})
    if not word_data:
        return None

    content = word_data.get("content", {})
    if not content:
        return None

    # --- 提取字段 ---

    # 1. 单词和等级
    word_head = word_entry.get("headWord", "")  # 从外层提取
    # 尝试从 bookId 或其他地方提取等级
    exam_level = (
        word_entry.get("bookId", "").split("_")[0] if word_entry.get("bookId") else ""
    )

    # 2. 中文释义 (meaning_zh)
    meaning_zh = ""
    trans_list = content.get("trans", [])
    if trans_list:
        # 提取第一个中文释义
        meaning_zh = trans_list[0].get("tranCn", "")

    # 3. 词性 (pos) 和同义词 (synonyms)
    pos_set = set()
    synonyms_list = []
    syno_data = content.get("syno", {})
    for item in syno_data.get("synos", []):
        if "pos" in item:
            pos_set.add(item["pos"].strip())
        for hw in item.get("hwds", []):
            synonyms_list.append(hw.get("w"))

    pos_str = ", ".join(sorted(list(pos_set)))  # 将词性去重并用逗号分隔
    synonyms_str = " | ".join(filter(None, synonyms_list))  # 将同义词用 | 分隔

    # 4. 同根词 (rel_words)
    rel_words_list = []
    rel_word_data = content.get("relWord", {})
    for rel_item in rel_word_data.get("rels", []):
        for w in rel_item.get("words", []):
            rel_words_list.append(w.get("hwd"))

    rel_words_str = " | ".join(filter(None, rel_words_list))

    # 5. 音标
    phonetic_uk = content.get("ukphone", "")
    phonetic_us = content.get("usphone", "")

    # 构造最终的字典
    return {
        "word": word_head,
        "meaning_zh": meaning_zh,
        "pos": pos_str,
        "phonetic_uk": phonetic_uk,
        "phonetic_us": phonetic_us,
        "synonyms": synonyms_str,
        "rel_words": rel_words_str,
        "exam_level": exam_level,
    }


def process_json_to_csv_fixed_v2():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到输入文件 '{INPUT_FILE}'。请确认文件路径。")
        return

    processed_data = []

    try:
        # 1. 逐行读取文件 (JSON Lines 格式)
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                try:
                    entry = json.loads(stripped_line)
                    result = extract_word_data(entry)

                    # 只有当成功提取到有效的 'word' 字段时才添加
                    if result and result.get("word"):
                        processed_data.append(result)

                except json.JSONDecodeError:
                    print(
                        f"警告：第 {line_number} 行解析失败，可能 JSON 格式错误或文件被截断。已跳过。"
                    )
                except Exception as e:
                    print(f"警告：处理第 {line_number} 行时发生未知错误: {e}")

        if not processed_data:
            print(
                "❌ 未提取到任何有效的单词数据。请检查 JSON 文件内容是否符合预期结构。"
            )

            # 如果解析到的行数很少，提示用户检查文件头尾是否完整
            if line_number < 10:
                print(
                    f"（文件总行数：{line_number}）如果文件行数很少，可能是文件被截断或内容为空。"
                )
            return

        # 2. 写入 CSV 文件
        # (写入逻辑保持不变)
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(processed_data)

        print(f"✅ 成功转换 {len(processed_data)} 个单词。")
        print(f"文件已保存至：{OUTPUT_FILE}")

    except UnicodeDecodeError:
        print("❌ 错误：文件编码问题。请尝试将文件编码更改为 UTF-8。")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")


if __name__ == "__main__":
    process_json_to_csv_fixed_v2()
