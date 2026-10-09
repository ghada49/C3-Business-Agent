"""Gradio deployment: run `python app.py` after configuring .env."""
import gradio as gr
from agent_core import BusinessAgent

agent = BusinessAgent()

# Gradio 6 uses OpenAI-style message history. safe_history in agent_core.py
# supports both plain text and Gradio's list-of-text-blocks content format.
demo = gr.ChatInterface(
    fn=agent.respond,
    chatbot=gr.Chatbot(height=440),
    title="CedarBridge Learning | Virtual Assistant",
    description=(
        "A fictional online STEM tutoring business. Ask about lessons and pricing, "
        "request follow-up, or leave feedback. Personal data is saved only to "
        "local demo logs with your permission."
    ),
    examples=[
        "What subjects do you teach and what are your prices?",
        "Are you open on Sundays?",
        "I want Math tutoring for Grade 10. How can someone contact me?",
        "I was unhappy with the scheduling. Please record my feedback.",
        "Do you offer SAT Chemistry?",
        "Who is your head of university admissions?",
    ],
    cache_examples=False,
    save_history=False,
)

if __name__ == "__main__":
    demo.launch()
