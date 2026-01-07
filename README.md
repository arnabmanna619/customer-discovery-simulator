# 🕵️ Customer Discovery Simulator

A Streamlit-based demo playground for entrepreneurship and product classes: Students can practice **customer discovery interviews** (inspired by *The Mom Test*) with a realistic **AI persona**, then end the session to receive **structured coaching feedback** on their interview technique.

**Live demo:** https://customer-discovery-simulator.streamlit.app/


## What this is

This app simulates a short customer interview where the “customer” is an AI agent **method-acting** a persona. Students define:

- **Customer segment**
- **Problem statement**
- **Hypothesis to test**
- (Optional) Generate a **persona** backstory

They then conduct the interview via chat and finish with **AI feedback** that critiques question quality, flags leading questions/pitching, and gives a grade with specific improvements.



## Why it’s useful

Customer discovery is a skill—and most people practice it for the first time on real customers. This tool helps students:

- Rehearse asking **past-focused**, behavioral questions
- Avoid **leading** or **hypothetical** questions that create false positives
- Learn to separate **research** from **pitching**
- Reflect on whether their “evidence” actually supports the hypothesis



## Key features

- **Provider switcher:** OpenAI or Google Gemini (via OpenAI-compatible endpoint)
- **Persona Builder (optional):** Generate a realistic persona grounded in a segment + problem
- **Interview mode:** Chat-based interview with natural, busy, human-like responses
- **End & Analyze:** “Professor-style” critique in Markdown + score out of 10
- **Export:** Download transcript + feedback as a `.txt` file
- **Streamlit Community Cloud friendly:** Works with `st.secrets` for classroom/demo keys



## How it works (high level)

1. A **system prompt** instructs the AI to inhabit a persona and follow *The Mom Test* logic:
   - Past experiences → specific, honest detail  
   - Future/hypotheticals → vague, non-committal  
   - Pitching detected → interest only if true “high pain,” otherwise polite indifference

2. At the end, the app compiles the transcript and sends it to a **coach prompt** that grades:
   - Question critique
   - Hypothesis verdict
   - Missed opportunities (workarounds, budget, frequency)
   - Score + actionable advice



## Quickstart (run locally)

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Run

```bash
streamlit run app.py
```

> If your main file is not `app.py`, update the command accordingly.



## Configuration

### Option A — Enter your API key in the sidebar

The app supports direct entry for:
- **OpenAI API Key**
- **Gemini API Key** (using the OpenAI-compatible base URL)

### Option B — Streamlit Secrets (recommended for demos/classrooms)

For the **Gemini (Test)** mode, the app reads:

```toml
# .streamlit/secrets.toml
GEMINI_TEST_API_KEY = "YOUR_KEY_HERE"
```

This is useful for a shared, rate-limited classroom key without distributing it in code.



## Providers & models

- **OpenAI** (default model configured in code)
- **Gemini** via OpenAI-compatible endpoint  
  - Base URL: `https://generativelanguage.googleapis.com/v1beta/openai/`



## Deployment

This app is deployed on **Streamlit Community Cloud**.

**Currenlty deployed at:** https://customer-discovery-simulator.streamlit.app/

## Disclaimer

- This is a **practice / rehearsal tool** for learning customer discovery technique—not a substitute for real interviews.
- AI personas can be inconsistent or wrong. Treat outputs as **training feedback**, not ground truth.
- Avoid entering sensitive, private, or regulated real-customer data into the app.



## Credits

Created at the **Wharton–Photon Startup AI Lab**  
Created by: **Arnab Manna, Hari Ravi, J. Daniel Kim (2026)**
