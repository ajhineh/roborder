import os
import logging
import asyncio
from typing import Dict, Literal, Optional
import ccxt.pro as ccxt

from src.config import Config

logger = logging.getLogger("ROBORDER.OrderExecutor")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class OrderExecutor:
    """
    موتور اجرای معاملات ناهمگام فیوچرز با استفاده از CCXT Pro.
    این کلاس مسئولیت ثبت اهرم داینامیک، ارسال سفارشات مارکت، مدیریت پوزیشن‌های باز،
    و اعمال کنترل‌های ریسک پیش‌معاملاتی (PTRC) را بر عهده دارد.
    در صورت عدم ارائه کلیدهای API صرافی، سیستم به صورت هوشمند روی شبیه‌ساز (Paper Trading) اجرا می‌شود.
    """
    def __init__(
        self,
        exchange_id: str = "binance",
        live_trading: bool = False,
        max_concurrent_positions: int = 3,       # حداکثر تعداد موقعیت‌های همزمان باز
        max_drawdown_limit_usdt: float = 100.0   # سقف حد ضرر روزانه حساب
    ):
        self.exchange_id = exchange_id
        self.live_trading = live_trading
        self.max_concurrent_positions = max_concurrent_positions
        self.max_drawdown_limit_usdt = max_drawdown_limit_usdt

        # کلیدهای صرافی از متغیرهای محیطی
        self.api_key = os.getenv("EXCHANGE_API_KEY", "")
        self.secret_key = os.getenv("EXCHANGE_SECRET_KEY", "")

        self.exchange: Optional[ccxt.Exchange] = None
        self.open_positions: Dict[str, dict] = {}
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        
        from datetime import datetime
        self.last_trade_date = datetime.now().strftime("%Y-%m-%d")

        self.setup_exchange()

    def _check_daily_reset(self) -> None:
        """بررسی تغییر روز برای بازنشانی دروداون و سود/زیان روزانه در ساعت ۲۴:۰۰"""
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        if not hasattr(self, "last_trade_date") or self.last_trade_date != current_date:
            self.daily_pnl = 0.0
            self.current_drawdown = 0.0
            self.last_trade_date = current_date
            logger.info("📅 New trading day detected! Resetting daily drawdown and PnL counters to $0.00.")

    def setup_exchange(self) -> None:
        """راه‌اندازی صرافی زنده یا حالت شبیه‌ساز معاملاتی"""
        if self.live_trading and self.api_key and self.secret_key:
            try:
                exchange_class = getattr(ccxt, self.exchange_id)
                self.exchange = exchange_class({
                    'apiKey': self.api_key,
                    'secret': self.secret_key,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',  # ورود به بازار USD-M Futures
                    }
                })
                logger.info(f"🟢 Connected to LIVE EXCHANGE: {self.exchange_id} (Futures USD-M mode active)")
            except Exception as e:
                logger.error(f"❌ Failed to connect to live exchange: {e}. Falling back to Simulation Mode.")
                self.live_trading = False
                self.exchange = None
        else:
            self.live_trading = False
            logger.info("🤖 Running in high-fidelity PAPER TRADING (Simulation Mode) - Zero risk to real funds")

    async def close_connections(self) -> None:
        """بستن ایمن اتصالات وب‌سوکت صرافی در زمان خروج ربات"""
        if self.exchange:
            await self.exchange.close()
            logger.info("🔌 Exchange connections closed.")

    async def execute_entry(
        self,
        symbol: str,
        side: Literal["long", "short"],
        amount_usdt: float,
        leverage: int,
        take_profit_quote: float,
        stop_loss_quote: float,
        entry_price: Optional[float] = None
    ) -> bool:
        """
        اجرای سفارش ورود مارکت به همراه تنظیم اهرم داینامیک و کنترل ریسک پیش‌معاملاتی (PTRC).
        """
        # ۱. کنترل ریسک پیش‌معاملاتی (PTRC)
        self._check_daily_reset()
        if len(self.open_positions) >= self.max_concurrent_positions:
            if "MAX_CONCURRENT_POSITIONS_CHECK" not in Config.BYPASSED_FILTERS_SET:
                logger.warning(f"⚠️ Blocked Entry for {symbol}: Maximum concurrent positions limit reached ({self.max_concurrent_positions})")
                return False

        if symbol in self.open_positions:
            logger.warning(f"⚠️ Blocked Entry: Position already open for {symbol}")
            return False

        if self.current_drawdown >= self.max_drawdown_limit_usdt:
            if "MAX_DRAWDOWN_LIMIT_CHECK" not in Config.BYPASSED_FILTERS_SET:
                logger.error(f"🚨 Blocked Entry: Daily account drawdown limit reached (${self.current_drawdown:.2f} >= ${self.max_drawdown_limit_usdt:.2f})")
                return False

        # کنترل موجودی مارجین آزاد حساب
        required_margin = (Config.TRADE_CAPITAL_PCT / 100.0) * Config.CURRENT_BALANCE
        used_margin = sum([pos.get("amount", 0.0) for pos in self.open_positions.values()])
        available_margin = Config.CURRENT_BALANCE - used_margin
        if available_margin < required_margin:
            if "AVAILABLE_MARGIN_CHECK" not in Config.BYPASSED_FILTERS_SET:
                logger.warning(f"⚠️ Blocked Entry for {symbol}: Insufficient Available Margin (${available_margin:.2f} available < required ${required_margin:.2f})")
                return False

        logger.info(f"⚡ PTRC Passed. Proceeding to execute entry order for {symbol} ({side.upper()}) | Size: ${amount_usdt} | Leverage: {leverage}x")

        # ۲. سناریو ترید زنده روی صرافی
        if self.live_trading and self.exchange:
            try:
                # الف. تنظیم اهرم معامله در صرافی
                await self.exchange.set_leverage(leverage, symbol)
                logger.info(f"✅ Set dynamic leverage to {leverage}x for {symbol} on exchange")

                # ب. تبدیل حجم معامله بر پایه USDT به مقدار توکن مورد نظر
                # برای مثال درPOP CAT ابتدا قیمت لحظه‌ای را دریافت می‌کنیم
                ticker = await self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                token_amount = (amount_usdt * leverage) / current_price

                # ج. ثبت سفارش مارکت صرافی
                order_side = "buy" if side == "long" else "sell"
                logger.info(f"🛒 Sending Market Order to Exchange: {order_side.upper()} {token_amount:.4f} {symbol}")
                
                order = await self.exchange.create_market_order(symbol, order_side, token_amount)
                logger.info(f"🎉 Trade Executed on Exchange. Order ID: {order['id']}")

                # ثبت موقعیت در پوزیشن‌های باز ربات
                self.open_positions[symbol] = {
                    "id": order["id"],
                    "side": side,
                    "leverage": leverage,
                    "entry_price": current_price,
                    "token_amount": token_amount,
                    "amount": amount_usdt,
                    "tp": take_profit_quote,
                    "sl": stop_loss_quote
                }
                return True

            except Exception as e:
                logger.error(f"❌ Exchange Execution Failed: {e}")
                return False

        # ۳. سناریو شبیه‌ساز (Paper Trading Mode)
        else:
            # شبیه‌سازی تاخیر صرافی (۵۰ میلی‌ثانیه)
            await asyncio.sleep(0.05)
            
            # استفاده از قیمت لحظه‌ای واقعی ثبت‌شده در معامله جهت محاسبه دقیق سود و زیان
            simulated_entry_price = entry_price if entry_price is not None else 1.0
            
            self.open_positions[symbol] = {
                "id": "mock_order_" + str(int(asyncio.get_event_loop().time() * 1000)),
                "side": side,
                "leverage": leverage,
                "entry_price": simulated_entry_price,
                "token_amount": (amount_usdt * leverage) / simulated_entry_price,
                "amount": amount_usdt,
                "tp": take_profit_quote,
                "sl": stop_loss_quote
            }
            logger.info(f"🎉 Simulated Trade Opened at ${simulated_entry_price:.6f}. Target SL: {stop_loss_quote:.6f} | Target TP: {take_profit_quote:.6f}")
            return True

    async def execute_exit(
        self,
        symbol: str,
        pnl_usdt: float,
        reason: str
    ) -> bool:
        """
        اجرای سفارش خروج مارکت برای بستن پوزیشن باز و به‌روزرسانی آمارهای روزانه ریسک.
        """
        self._check_daily_reset()
        position = self.open_positions.get(symbol)
        if not position:
            logger.warning(f"⚠️ No active open position found to exit for {symbol}")
            return False

        logger.info(f"⚡ Initiating exit order for {symbol} | Reason: {reason} | Expected PnL: ${pnl_usdt:+.4f} USDT")

        from src.config import Config, save_env_values
        is_partial = (reason == "TP1")
        tp1_exit_fraction = getattr(Config, "TP1_EXIT_PCT", 50.0) / 100.0

        # ۱. سناریو ترید زنده روی صرافی با پیاده‌سازی تعقیب لیمیت PostOnly و فالبک اضطراری به مارکت اردر
        if self.live_trading and self.exchange:
            try:
                exit_side = "sell" if position["side"] == "long" else "buy"
                token_amount = position["token_amount"]
                if is_partial:
                    token_amount *= tp1_exit_fraction

                logger.info(f"🛒 Initiating PostOnly Limit Exit for {symbol} | Amount: {token_amount:.4f}")
                
                # دریافت قیمت بهترین بید/اسک جهت قرارگیری به صورت Maker
                ticker = await self.exchange.fetch_ticker(symbol)
                limit_price = ticker['bid'] if exit_side == "sell" else ticker['ask']
                
                # ثبت سفارش لیمیت PostOnly اول
                order = await self.exchange.create_limit_order(
                    symbol=symbol,
                    side=exit_side,
                    amount=token_amount,
                    price=limit_price,
                    params={"postOnly": True}
                )
                order_id = order['id']
                logger.info(f"⚡ Placed initial PostOnly Limit Exit order {order_id} at ${limit_price:.6f}")

                # شروع لوپ تعقیب سفارش (Order Chasing) برای مدت حداکثر ۱۰ ثانیه
                start_time = asyncio.get_event_loop().time()
                chase_timeout = 10.0
                poll_interval = 2.0
                is_filled = False

                while (asyncio.get_event_loop().time() - start_time) < chase_timeout:
                    await asyncio.sleep(poll_interval)
                    
                    # بررسی وضعیت سفارش در صرافی
                    try:
                        order_status = await self.exchange.fetch_order(order_id, symbol)
                        status = order_status.get('status')
                        filled_amount = order_status.get('filled', 0.0)
                        remaining_amount = token_amount - filled_amount
                        
                        if status == 'closed' or remaining_amount <= 0:
                            logger.info(f"🎉 Limit Exit order {order_id} fully filled!")
                            is_filled = True
                            break
                        
                        # اگر سفارش به دلیلی لغو شد (مانند رد شدن توسط postOnly)، سفارش لیمیت جدید ثبت می‌شود
                        if status == 'canceled':
                            logger.warning(f"⚠️ Order {order_id} was canceled. Re-submitting limit order...")
                            ticker = await self.exchange.fetch_ticker(symbol)
                            limit_price = ticker['bid'] if exit_side == "sell" else ticker['ask']
                            order = await self.exchange.create_limit_order(
                                symbol=symbol,
                                side=exit_side,
                                amount=remaining_amount,
                                price=limit_price,
                                params={"postOnly": True}
                            )
                            order_id = order['id']
                            continue
                            
                    except Exception as e:
                        logger.error(f"Error polling order {order_id}: {e}")
                        continue
                        
                    # بررسی تغییر بید/اسک مارکت و تعقیب قیمت (Chasing)
                    ticker = await self.exchange.fetch_ticker(symbol)
                    current_market_price = ticker['bid'] if exit_side == "sell" else ticker['ask']
                    
                    # اگر قیمت مارکت جابجا شده باشد
                    if current_market_price != limit_price:
                        logger.info(f"🔄 Price moved from ${limit_price:.6f} to ${current_market_price:.6f}. Chasing order...")
                        try:
                            # لغو سفارش قبلی
                            await self.exchange.cancel_order(order_id, symbol)
                        except Exception as cancel_err:
                            logger.warning(f"Failed to cancel order {order_id}: {cancel_err}")
                        
                        # قرار دادن سفارش جدید روی قیمت جدید
                        limit_price = current_market_price
                        try:
                            order = await self.exchange.create_limit_order(
                                symbol=symbol,
                                side=exit_side,
                                amount=remaining_amount,
                                price=limit_price,
                                params={"postOnly": True}
                            )
                            order_id = order['id']
                            logger.info(f"⚡ Re-placed PostOnly Limit Exit order {order_id} at ${limit_price:.6f}")
                        except Exception as place_err:
                            logger.error(f"Failed to place chased order: {place_err}")

                # ۴. فالبک اضطراری به سفارش مارکت در صورت عدم پر شدن پس از ۱۰ ثانیه
                if not is_filled:
                    logger.warning(f"🚨 Limit Exit Chase timed out after {chase_timeout}s! Executing Emergency Market Fallback...")
                    try:
                        # تلاش برای لغو سفارش لیمیت باز باقیمانده
                        await self.exchange.cancel_order(order_id, symbol)
                    except Exception as e:
                        logger.warning(f"Failed to cancel final limit order {order_id}: {e}")
                    
                    # استعلام آخرین مانده پر نشده سفارش
                    try:
                        order_status = await self.exchange.fetch_order(order_id, symbol)
                        remaining_amount = token_amount - order_status.get('filled', 0.0)
                    except Exception:
                        remaining_amount = token_amount # فالبک در صورت خطا
                    
                    if remaining_amount > 0:
                        logger.info(f"🛒 Sending Emergency Market Exit Order (Limit with 2% Slippage Guard) for {symbol} | Amount: {remaining_amount:.4f}")
                        try:
                            # دریافت قیمت بهترین بید/اسک زنده جهت اعمال انحراف ۲٪ به عنوان سقف لغزش قیمت
                            ticker = await self.exchange.fetch_ticker(symbol)
                            current_market_price = ticker['bid'] if exit_side == "sell" else ticker['ask']
                            
                            # اعمال ۲٪ انحراف قیمت (در پوزیشن خرید جهت فروش پایین‌تر، در پوزیشن فروش جهت خرید بالاتر)
                            if exit_side == "sell":
                                protect_price = current_market_price * 0.98
                            else:
                                protect_price = current_market_price * 1.02
                                
                            # ارسال سفارش لیمیت عادی (بدون postOnly) با قیمت محافظت شده جهت تضمین پر شدن سریع
                            order = await self.exchange.create_limit_order(
                                symbol=symbol,
                                side=exit_side,
                                amount=remaining_amount,
                                price=protect_price
                            )
                            logger.info(f"🎉 Emergency Exit Limit Order with 2% Slippage Guard executed. Order ID: {order['id']}")
                        except Exception as market_err:
                            logger.error(f"Slippage Guard Limit Order failed: {market_err}. Attempting raw market order fallback...")
                            # فالبک به سفارش مارکت خالص در صورت بروز خطای لیمیت اردر
                            order = await self.exchange.create_market_order(symbol, exit_side, remaining_amount)
                            logger.info(f"🎉 Emergency Raw Market Exit executed. Order ID: {order['id']}")

            except Exception as e:
                logger.error(f"❌ Exchange Exit Order Pipeline Failed: {e}. Positions will be forcefully closed locally.")

        # ۲. اعمال تغییرات سود و ضرر در ردیاب دروداون روزانه بر اساس کل سود/زیان خالص روزانه
        self.daily_pnl += pnl_usdt
        if self.daily_pnl >= 0:
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = abs(self.daily_pnl)

        # به‌روزرسانی و ذخیره‌سازی ماندگار موجودی کل حساب (CURRENT_BALANCE)
        new_balance = Config.CURRENT_BALANCE + pnl_usdt
        Config.CURRENT_BALANCE = new_balance
        save_env_values({"CURRENT_BALANCE": f"{new_balance:.4f}"})

        # به‌روزرسانی یا پاک‌سازی موقعیت از حافظه محلی
        if is_partial:
            if symbol in self.open_positions:
                self.open_positions[symbol]["amount"] *= (1.0 - tp1_exit_fraction)
                self.open_positions[symbol]["token_amount"] *= (1.0 - tp1_exit_fraction)
            logger.info(f"🛡️ Position for {symbol} partially closed ({getattr(Config, 'TP1_EXIT_PCT', 50.0)}%). Remaining amount: ${self.open_positions[symbol]['amount']:.2f}")
        else:
            if symbol in self.open_positions:
                del self.open_positions[symbol]
            logger.info(f"🚪 Position Fully Closed for {symbol}. Current Daily Drawdown: ${self.current_drawdown:.2f} | New Balance: ${new_balance:.4f}")

        return True
