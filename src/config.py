import os
import logging
from typing import Dict, List, Literal

logger = logging.getLogger("ROBORDER.Config")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


ENV_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))


def load_env_file(filepath: str = ENV_FILE_PATH) -> None:
    """
    پارس کردن بومی فایل متغیرهای محیطی (.env) بدون نیاز به کتابخانه dotenv.
    این تابع مقادیر کلید-مقدار را خوانده و در os.environ بارگذاری می‌کند.
    """
    if not os.path.exists(filepath):
        logger.warning(f"⚠️ Configuration file not found at {filepath}. Using defaults or system environment variables.")
        return

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # نادیده گرفتن خطوط خالی و کامنت‌ها
                if not line or line.startswith("#"):
                    continue
                
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    # حذف کامنت‌های داخل خط
                    if "#" in value:
                        value = value.split("#", 1)[0]
                    value = value.strip()
                    
                    # نادیده گرفتن مقادیر داخل کوتیشن
                    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    
                    # تنها در صورتی ست می‌شود که قبلاً در سیستم تعریف نشده باشد
                    if key: #and key not in os.environ:
                        os.environ[key] = value
        logger.info(f"📂 Successfully loaded and normalized configurations from {filepath}")
    except Exception as e:
        logger.error(f"Failed to parse .env file: {e}")


# اجرای پیش‌فرض لود کردن متغیرها از ریشه پروژه
load_env_file(ENV_FILE_PATH)


class Config:
    """کلاس یکپارچه و تایپ‌شده برای ارائه متغیرهای تنظیمی ربات ROBORDER-X"""
    
    # --- عمومی ---
    EXCHANGE_ID: str = "binance"
    ROBORDER_LIVE: bool = False
    EXCHANGE_API_KEY: str = ""
    EXCHANGE_SECRET_KEY: str = ""
    QUOTE_DENOMINATION: Literal["USDT", "SOL"] = "USDT"
    CUSTOM_WS_ENDPOINT: str = ""
    HELIUS_WS_URL: str = ""
    QUICKNODE_WS_URL: str = ""
    HISTORY_FILE_PATH: str = "microtick_history.json"
    MACRO_NEWS_SCHEDULE_FILE: str = "macro_news_schedule.json"
    CLOCK_DRIFT_MS: int = 0
    SYMBOLS: List[str] = []

    # --- دفترچه سفارشات LOB ---
    LOB_DEPTH_LEVELS: int = 5
    TRADE_WINDOW_SECONDS: int = 10
    SPOOF_THRESHOLD_PCT: float = 0.15

    # --- استراتژی مومنتوم اسکلپر ---
    MOMENTUM_WINDOW_MS: int = 10000
    SPREAD_THRESHOLD_BPS: int = 8
    TAKE_PROFIT_BPS: int = 15
    STOP_LOSS_BPS: int = 10
    COOLDOWN_MS: int = 30000

    # --- مدیریت ریسک PTRC ---
    MAX_CONCURRENT_POSITIONS: int = 2
    MAX_DRAWDOWN_LIMIT_USDT: float = 100.0
    INITIAL_BALANCE: float = 10000.0
    CURRENT_BALANCE: float = 10000.0
    TRADE_CAPITAL_PCT: float = 10.0
    USE_ONLY_PPO: bool = False
    USE_YOYO_STRATEGY: bool = True
    YOYO_RISK_PCT: float = 1.0
    DEFAULT_LEVERAGE: int = 15
    MAX_LEVERAGE: int = 25
    BYPASSED_FILTERS: str = ""
    BYPASSED_FILTERS_SET: set = set()

    @classmethod
    def reload(cls) -> None:
        """بارگذاری مجدد و داینامیک تمام تنظیمات در حافظه موقت"""
        # پاک‌سازی متغیرهای محیطی بارگذاری شده قبلی تا تغییرات اعمال شوند
        keys_to_clear = [
            "EXCHANGE_ID", "ROBORDER_LIVE", "EXCHANGE_API_KEY", "EXCHANGE_SECRET_KEY",
            "QUOTE_DENOMINATION", "CUSTOM_WS_ENDPOINT", "HELIUS_WS_URL", "QUICKNODE_WS_URL",
            "HISTORY_FILE_PATH", "SYMBOLS", "LOB_DEPTH_LEVELS", "TRADE_WINDOW_SECONDS",
            "SPOOF_THRESHOLD_PCT", "MOMENTUM_WINDOW_MS", "SPREAD_THRESHOLD_BPS",
            "TAKE_PROFIT_BPS", "STOP_LOSS_BPS", "COOLDOWN_MS", "MAX_CONCURRENT_POSITIONS",
            "MAX_DRAWDOWN_LIMIT_USDT", "INITIAL_BALANCE", "CURRENT_BALANCE", "TRADE_CAPITAL_PCT",
            "USE_ONLY_PPO", "USE_YOYO_STRATEGY", "YOYO_RISK_PCT", "DEFAULT_LEVERAGE", "MAX_LEVERAGE", "BYPASSED_FILTERS"
        ]
        for key in keys_to_clear:
            os.environ.pop(key, None)
            
        load_env_file(ENV_FILE_PATH)
        
        cls.EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")
        cls.ROBORDER_LIVE = os.getenv("ROBORDER_LIVE", "false").lower() == "true"
        cls.EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY", "")
        cls.EXCHANGE_SECRET_KEY = os.getenv("EXCHANGE_SECRET_KEY", "")
        cls.QUOTE_DENOMINATION = os.getenv("QUOTE_DENOMINATION", "USDT") # type: ignore
        cls.CUSTOM_WS_ENDPOINT = os.getenv("CUSTOM_WS_ENDPOINT", "")
        cls.HELIUS_WS_URL = os.getenv("HELIUS_WS_URL", "")
        cls.QUICKNODE_WS_URL = os.getenv("QUICKNODE_WS_URL", "")
        cls.HISTORY_FILE_PATH = os.getenv("HISTORY_FILE_PATH", "microtick_history.json")

        raw_symbols = os.getenv("SYMBOLS", "POPCAT/USDT:USDT,WIF/USDT:USDT,BOME/USDT:USDT,SOL/USDT:USDT")
        cls.SYMBOLS = [sym.strip() for sym in raw_symbols.split(",") if sym.strip()]

        cls.LOB_DEPTH_LEVELS = int(os.getenv("LOB_DEPTH_LEVELS", "5"))
        cls.TRADE_WINDOW_SECONDS = int(os.getenv("TRADE_WINDOW_SECONDS", "10"))
        cls.SPOOF_THRESHOLD_PCT = float(os.getenv("SPOOF_THRESHOLD_PCT", "0.15"))

        cls.MOMENTUM_WINDOW_MS = int(os.getenv("MOMENTUM_WINDOW_MS", "10000"))
        cls.SPREAD_THRESHOLD_BPS = int(os.getenv("SPREAD_THRESHOLD_BPS", "8"))
        cls.TAKE_PROFIT_BPS = int(os.getenv("TAKE_PROFIT_BPS", "15"))
        cls.STOP_LOSS_BPS = int(os.getenv("STOP_LOSS_BPS", "10"))
        cls.COOLDOWN_MS = int(os.getenv("COOLDOWN_MS", "30000"))

        cls.MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "2"))
        cls.MAX_DRAWDOWN_LIMIT_USDT = float(os.getenv("MAX_DRAWDOWN_LIMIT_USDT", "100.0"))
        cls.INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "10000.0"))
        cls.CURRENT_BALANCE = float(os.getenv("CURRENT_BALANCE", "10000.0"))
        cls.TRADE_CAPITAL_PCT = float(os.getenv("TRADE_CAPITAL_PCT", "10.0"))
        cls.USE_ONLY_PPO = os.getenv("USE_ONLY_PPO", "false").lower() == "true"
        cls.USE_YOYO_STRATEGY = not cls.USE_ONLY_PPO
        cls.YOYO_RISK_PCT = float(os.getenv("YOYO_RISK_PCT", "1.0"))
        cls.DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "15"))
        cls.MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", "25"))
        cls.BYPASSED_FILTERS = os.getenv("BYPASSED_FILTERS", "")
        cls.BYPASSED_FILTERS_SET = {f.strip() for f in cls.BYPASSED_FILTERS.split(",") if f.strip()}
        logger.info("🔄 Config properties reloaded successfully.")


# اجرای لود اولیه کلاس متغیرها
Config.reload()


def save_env_values(updates: Dict[str, str], filepath: str = ENV_FILE_PATH) -> bool:
    """
    بروزرسانی در لحظه مقادیر فایل .env بدون حذف توضیحات فارسی و فرمت‌بندی خطوط.
    این تابع فایل را خط به خط خوانده، مقادیر متناظر را ویرایش کرده و تغییرات را ذخیره می‌کند.
    """
    if not os.path.exists(filepath):
        logger.error(f"Cannot save settings: .env file does not exist at {filepath}")
        return False

    try:
        updated_lines = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    updated_lines.append(line)
                    continue

                key, val = stripped.split("=", 1)
                key = key.strip()
                
                if key in updates:
                    # پیدا کردن کامنت‌های داخل خط در صورت وجود جهت حفظ آن‌ها
                    comment = ""
                    if "#" in val:
                        val_part, comment_part = val.split("#", 1)
                        comment = "  #" + comment_part
                    
                    new_val = updates[key]
                    # نوشتن خط جدید با قالب‌بندی اصلی و کامنت حفظ‌شده
                    new_line = f"{key}={new_val}{comment}\n"
                    updated_lines.append(new_line)
                    # حذف کلید از دیکشنری بروزرسانی‌ها جهت نشانه‌گذاری به عنوان انجام شده
                    del updates[key]
                else:
                    updated_lines.append(line)

        # اگر کلیدهای جدیدی وجود دارند که در فایل نبودند، آن‌ها را به انتهای فایل الحاق می‌کنیم
        if updates:
            updated_lines.append("\n# --- تنظیمات الحاق شده داینامیک از سمت وب --- \n")
            for key, val in updates.items():
                updated_lines.append(f"{key}={val}\n")

        # ذخیره کل خطوط به درون فایل اصلی
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(updated_lines)

        logger.info(f"💾 Updated {len(updated_lines)} lines in .env file successfully.")
        
        # بارگذاری مجدد تنظیمات در حافظه موقت برنامه
        Config.reload()
        return True

    except Exception as e:
        logger.error(f"Failed to save environment values: {e}")
        return False
