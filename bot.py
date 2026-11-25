import asyncio
import logging
import os
from collections import defaultdict
from typing import DefaultDict, List, Dict

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip()
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful Telegram assistant. Keep answers clear and concise.",
).strip()
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in environment.")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment.")

client = OpenAI(api_key=OPENAI_API_KEY)

# In-memory chat history per Telegram user.
# Good enough for local development; use Redis or a database in production.
chat_history: DefaultDict[int, List[Dict[str, str]]] = defaultdict(list)


def build_openai_input(user_id: int, user_text: str) -> List[Dict[str, str]]:
    history = chat_history[user_id][-MAX_HISTORY_MESSAGES:]
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


def ask_openai(user_id: int, user_text: str) -> str:
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=build_openai_input(user_id, user_text),
    )
    answer = (response.output_text or "").strip()

    if not answer:
        answer = "I couldn't generate a reply just now. Please try again."

    chat_history[user_id].append({"role": "user", "content": user_text})
    chat_history[user_id].append({"role": "assistant", "content": answer})

    # Keep memory bounded.
    if len(chat_history[user_id]) > MAX_HISTORY_MESSAGES * 2:
        chat_history[user_id] = chat_history[user_id][-(MAX_HISTORY_MESSAGES * 2) :]

    return answer


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    await update.message.reply_text(
        "Hi! Send me a message and I'll reply using OpenAI.\n\n"
        "Use /reset to clear our chat memory."
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.message:
        return

    chat_history.pop(update.effective_user.id, None)
    await update.message.reply_text("Chat memory cleared.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    await update.message.reply_text(
        "Commands:\n"
        "/start - start the bot\n"
        "/help - show help\n"
        "/reset - clear conversation history"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    user_text = update.message.text.strip()

    if not user_text:
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        answer = await asyncio.to_thread(ask_openai, user_id, user_text)
        await update.message.reply_text(answer)
    except Exception:
        logger.exception("Failed to process message")
        await update.message.reply_text(
            "Something went wrong while talking to the AI service."
        )


def build_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return app


def main() -> None:
    logger.info("Starting Telegram bot with model %s", OPENAI_MODEL)
    app = build_app()
    app.run_polling()


if __name__ == "__main__":
    main()
