from datetime import datetime
from sqlalchemy import UniqueConstraint
from .db_instance import db  # ✅ 导入 SQLAlchemy 实例
from typing import Dict, Any


class WordState(db.Model):
    """
    用户单词记忆状态模型，基于 SM2 算法参数设计。
    """

    __tablename__ = "word_states"
    # 简化：使用 default=1 作为默认用户ID
    USER_ID = 1

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, default=USER_ID, nullable=False)
    word = db.Column(db.String(100), nullable=False)

    # SM2 状态
    easiness_factor = db.Column(db.Float, default=2.5)  # 熟练度因子 (EF)
    repetitions = db.Column(db.Integer, default=0)  # 重复次数 (n)
    interval = db.Column(db.Integer, default=0)  # 间隔天数 (I)
    # 下次复习日期
    next_review_date = db.Column(db.DateTime, default=datetime.now)

    # 确保 word + user_id 唯一
    __table_args__ = (UniqueConstraint("user_id", "word", name="_user_word_uc"),)

    def to_dict(self, meaning_zh: str = "（含义缺失）") -> Dict[str, Any]:
        """将模型对象转换为可序列化的字典，用于 API 响应"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "word": self.word,
            # 含义由外部逻辑（如 app.py 或 memory_logic.py）传入，避免循环导入
            "meaning_zh": meaning_zh,
            "easiness_factor": round(self.easiness_factor, 2),  # 保持两位小数
            "repetitions": self.repetitions,
            "interval": self.interval,
            "next_review_date": self.next_review_date.strftime("%Y-%m-%d %H:%M:%S"),
            "is_due": (
                self.next_review_date.date() <= datetime.now().date()
            ),  # 添加一个复习提醒标志
        }

    def __repr__(self):
        return f"<WordState word='{self.word}' user_id={self.user_id} next_review={self.next_review_date.date()}>"
