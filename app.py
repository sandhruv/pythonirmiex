import os

import gradio as gr
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer clearly, accurately, and concisely."
)
EXAMPLES = [
    "Explain quantum computing in simple terms.",
    "Give me a 3-day itinerary for Paris.",
    "Write a Python function that reverses a string.",
]

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None


def content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, list):
        parts = [content_to_text(item) for item in content]
        return "\n".join(part for part in parts if part)
    return str(content) if content is not None else ""


def chat(message: str, history: list[dict]) -> str:
    if client is None:
        return (
            "The server is missing its Groq API key. "
            "Add GROQ_API_KEY to the Render environment variables."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for item in history:
        role = item.get("role")
        if role in ("user", "assistant"):
            text = content_to_text(item.get("content"))
            if text:
                messages.append({"role": role, "content": text})

    messages.append({"role": "user", "content": message})

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_completion_tokens=1024,
        )
        return (
            completion.choices[0].message.content
            or "I don't have a response for that."
        )
    except Exception:
        return "I couldn't contact Groq. Please try again in a moment."


demo = gr.ChatInterface(
    fn=chat,
    title="Groq AI Chatbot",
    description="Ask me anything. Powered by Groq.",
    examples=EXAMPLES,
    flagging_mode="never",
    save_history=True,
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        ssr_mode=False,
    )
