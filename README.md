# CedarBridge Learning — Agent-Powered Business Assistant
## Ghada Al Danab

A  Gradio chatbot for a Lebanese online tutoring business. Uses OpenAI Chat Completions, an LLM-powered three-way router, function tools for lead collection and feedback logging, short-term conversation memory, and an optional reflection step.

## Quick start (Windows PowerShell)

From the `C3-Business-Agent` repository folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and **replace the example API key** with your own OpenAI API key. Choose an accessible model, e.g. `gpt-4.1-mini`. API calls incur charges. Never upload the `.env` file to GitHub.

Then run:

```powershell
python app.py
```

Open the local URL printed by Gradio, typically `http://127.0.0.1:7860`. To use the notebook instead, run:

```powershell
python -m pip install jupyterlab
jupyter lab business_agent.ipynb
```

Run notebook cells in order, from the project folder. In Colab, upload or clone the *whole* repository first and change the working directory to the folder; the PDF/TXT must be present.

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with a real OpenAI key
python app.py
```

## How the system works

```
customer message + short-term chat history
      |
      v
LLM router: general_question / lead_collection / feedback_unknown
      |
      v
LLM agent: persona + instructions + PDF + TXT + tool schemas
      |
      +--> general answer grounded in business facts
      +--> record_customer_interest(email, name, message) -> logs/leads.jsonl
      +--> record_feedback(question) -> logs/feedback.jsonl
      |
      v
(optional) second-model reflection -> logs/reflections.jsonl
      |
      v
Gradio chatbot reply
```

The router is a small LLM classification call, not a keyword-only simulation. The model chooses the permitted function tool, and Python validates the arguments, executes the function, sends the result back to the model, and then returns the final message. For a `feedback_unknown` classification, the model is required to call `record_feedback` before replying.

**Privacy:** Lead logging requires *explicit permission* and valid name/email/message. Logs are local, gitignored, and not sent to a real business. A publicly hosted Spaces demo uses an ephemeral filesystem by default; local logs are **not persistent CRM storage**. Use fictional contact details when recording the course demonstration.

## Demo messages

1. **General question:** `What subjects do you teach and what are your prices?` → general route with sourced answer.
2. **Hours:** `Are you open on Sundays?` → general route, factual response.
3. **Incomplete lead:** `I want Grade 10 Math tutoring.` → lead route, asks for name, email, and permission.
4. **Complete lead with permission:** `My name is Maya, my email is maya@example.com. Please contact me about Grade 10 Math tutoring.` → model calls `record_customer_interest` → new row in `logs/leads.jsonl`.
5. **Complaint:** `I am unhappy with the scheduling. Please record my feedback.` → model calls `record_feedback`.
6. **Unanswered question:** `Who is your head of university admissions?` → feedback/unknown route → log in `logs/feedback.jsonl`; no invented employee.

Console output includes `[router] ...`, `[tool] ...`, and (when enabled) `[reflection] ...`. Tool results are written to `.jsonl` files, one JSON record per line. **The example `.example` email addresses are intentionally non-deliverable**.

**The LLM can occasionally misroute or respond imperfectly.** Run the demo cases above with your actual API key and model to verify routing, replies, and tool calls. The chatbot does not really book lessons or send emails.

## Optional reflection bonus

In `.env`, set:

```
REFLECTION_ENABLED=true
```

For each final answer, the agent asks the LLM to review the draft for unsupported claims or instruction violations, then logs both the draft and revised response to `logs/reflections.jsonl`. This is an extra model call and will increase cost and response time.


## Deploy to Hugging Face Spaces

1. Create a new Space with **Gradio** as the SDK and **Python**.
2. Upload the project (without `.env`, `logs/*.jsonl`, local virtualenvs or keys).
3. Keep the Space-generated `README.md` with its Gradio metadata (`sdk: gradio`, `app_file: app.py`); do not overwrite it with this repository's `README.md`.
4. In the Space **Settings → Variables and secrets**, add a secret named `OPENAI_API_KEY`; set `OPENAI_MODEL` and `REFLECTION_ENABLED` as ordinary variables if you wish.
5. Spaces should run `app.py`; wait for the build and confirm the sample prompts work. Logs on free Spaces are not durable.

Do not expose personal contact information on a public Space. For the class demo, use synthetic data and explain it is simulated.
