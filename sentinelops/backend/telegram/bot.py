"""
Telegram Alert Bot (L4)
------------------------
Sends alerts to on-duty engineers when urgency is HIGH or CRITICAL.
In production, also integrate with Huawei SMSMSG for SMS fallback.

Run separately: python telegram/bot.py
Or integrate into the FastAPI lifespan for inline alerting.
"""
import asyncio
import os
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")


async def send_alert(message: str) -> None:
    """Send a work order alert to the configured Telegram chat."""
    if not BOT_TOKEN or not CHAT_ID:
        print("[Telegram] BOT_TOKEN or CHAT_ID not configured — skipping alert")
        return
    bot = Bot(token=BOT_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=f"🚨 SentinelOps Alert\n\n{message}", parse_mode="Markdown")


# ── Bot command handlers ────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 SentinelOps Bot online.\n"
        "I will notify you when machines require urgent attention.\n\n"
        "Commands: /status /help"
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ SentinelOps system online. All agents operational.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Use /status to check system health.")


if __name__ == "__main__":
    if not BOT_TOKEN:
        print("Set TELEGRAM_BOT_TOKEN in .env to run the bot.")
    else:
        app = ApplicationBuilder().token(BOT_TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("status", status))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
        print("Telegram bot polling…")
        app.run_polling()
