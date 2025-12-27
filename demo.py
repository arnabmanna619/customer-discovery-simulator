import streamlit as st
from openai import OpenAI

# ======================================================================================
# PAGE CONFIGURATION
# ======================================================================================
st.set_page_config(page_title="Customer Discovery Simulator", page_icon="🕵️", layout="wide")

# ======================================================================================
# CONSTANTS
# ======================================================================================
PROVIDERS = ("OpenAI", "Gemini", "Gemini (Test)")
DEFAULT_OPENAI_MODEL = "gpt-5.1"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_OPENAI_COMPAT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

HELP_TEXT = (
    "**Instructions:**\n"
    "1. Fill in your hypothesis details.\n"
    "2. (Optional) Generate a specific persona.\n"
    "3. Click 'Start Interview' to chat with the AI.\n"
    "4. Click 'End & Analyze' to get graded."
)

# ======================================================================================
# SESSION STATE INITIALIZATION
# ======================================================================================
def init_session_state() -> None:
    """Initialize all session state keys used by the app."""
    defaults = {
        "messages": [],
        "interview_active": False,
        "generated_persona": "",
        "analysis_done": False,
        "feedback_text": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_simulation() -> None:
    """Reset the simulation to its initial state and rerun."""
    st.session_state.messages = []
    st.session_state.interview_active = False
    st.session_state.generated_persona = ""
    st.session_state.analysis_done = False
    st.session_state.feedback_text = ""
    st.rerun()


init_session_state()

# ======================================================================================
# LLM HELPERS
# ======================================================================================
def escape_dollars(stream):
    """
    Streamlit Markdown treats $ specially (LaTeX).
    Escape dollar signs in streamed content to avoid accidental math rendering.
    """
    for event in stream:
        delta = event.choices[0].delta
        text = getattr(delta, "content", None)
        if text:
            yield text.replace("$", r"\$")


def build_client(api_key: str, base_url: str | None) -> OpenAI:
    """Create an OpenAI-compatible client (OpenAI or Gemini via OpenAI compatibility layer)."""
    return OpenAI(api_key=api_key, base_url=base_url)


def get_provider_config(provider: str):
    """
    Return (api_key, base_url, model_name) based on provider selection and UI inputs.
    Business logic preserved exactly: OpenAI uses default base URL; Gemini uses OpenAI-compatible base URL; Gemini Test uses st.secrets.
    """
    api_key: str = ""
    base_url: str | None = None
    model_name: str = ""

    if provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/docs/quickstart)"
        base_url = None
        model_name = DEFAULT_OPENAI_MODEL

    elif provider == "Gemini":
        api_key = st.text_input("Gemini API Key", type="password")
        "[Get Gemini API key](https://ai.google.dev/gemini-api/docs/api-key)"
        base_url = GEMINI_OPENAI_COMPAT_BASE_URL
        model_name = DEFAULT_GEMINI_MODEL

    else:  # Gemini (Test)
        st.success("Using a Free Test Key. This has limits.")
        api_key = st.secrets.get("GEMINI_TEST_API_KEY", "")
        base_url = GEMINI_OPENAI_COMPAT_BASE_URL
        model_name = DEFAULT_GEMINI_MODEL

    return api_key, base_url, model_name


def build_system_prompt(problem_statement: str, hypothesis_to_validate: str, persona_context: str) -> str:
    """Construct the system prompt used to keep the assistant in character during the interview."""
    return f"""
I want you to act as a specific persona for a mock customer discovery interview. Do not break character until I say "STOP INTERVIEW."

**My Business Context:**
I am investigating a problem related to: {problem_statement}.
My Hypothesis is: {hypothesis_to_validate}.

**Your Persona:**
{persona_context}

**Rules for your Roleplay:**
1. Be realistic. Real people are busy and sometimes indifferent.
2. If I ask a "Yes/No" question, give a short answer.
3. If I ask a "Leading Question" (e.g., "Wouldn't you love an app that...?"), react with skepticism or polite disinterest.
4. If I ask a great "Open-Ended Question" (e.g., "Tell me about the last time you...?"), open up and share details.
5. DO NOT offer solutions. Only talk about your life, problems, and current behaviors.
""".strip()


def build_persona_prompt(customer_segment: str, problem_statement: str) -> str:
    """Prompt used to generate a short persona backstory."""
    return (
        f"I am an entrepreneur. My customer segment is: {customer_segment}. "
        f"The problem they face is: {problem_statement}. "
        "Create a brief, specific 3-line persona description for a mock interview. "
        "Include a specific name, major/job, and a hidden 'ground truth' about how they currently solve this problem."
    )


def build_coach_prompt(transcript: str) -> str:
    """Prompt used to produce professor-style feedback on the interview transcript."""
    return f"""
You are an Expert Entrepreneurship Professor (like the one teaching MGMT 801 at Wharton).
Please analyze the transcript of the student's mock interview below.

TRANSCRIPT:
{transcript}

Provide feedback in markdown format:
1. **Question Quality:** Did the student ask "Leading Questions" (bad) or "Open-Ended Questions" (good)? Quote a specific mistake if present.
2. **Hypothesis Validation:** Based ONLY on the persona's answers, was the hypothesis validated or invalidated?
3. **Missing Info:** What is one critical thing they failed to ask?
4. **Grading:** Give a score out of 10 for unbiased interviewing technique.
""".strip()


def transcript_from_messages(messages: list[dict]) -> str:
    """Create a plain-text transcript for analysis (excluding system messages)."""
    return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages if m["role"] != "system"])


# ======================================================================================
# SIDEBAR: CONFIGURATION
# ======================================================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    selected_provider = st.selectbox("Select LLM Provider", PROVIDERS)
    api_key, base_url, model_name = get_provider_config(selected_provider)

    st.info(HELP_TEXT)

    if st.button("New Simulation", icon=":material/playlist_add:", type="secondary"):
        reset_simulation()

# ======================================================================================
# MAIN PAGE: HEADER
# ======================================================================================
st.title("🕵️ Customer Discovery Simulator")
st.markdown("Sharpen your customer discovery skills before talking to real customers.")

# ======================================================================================
# SECTION 1: INPUTS (only shown before interview starts)
# ======================================================================================
if not st.session_state.interview_active:
    col_left, col_right = st.columns(2)

    with col_left:
        problem_statement = st.text_area(
            "Problem Statement",
            placeholder="e.g., College students spend too much money buying formal wear for one-time events.",
            height=100,
        )

    with col_right:
        customer_segment = st.text_area(
            "Customer Segment",
            placeholder="e.g., A Sophomore at UPenn who attends 3-4 formals a year but is on a tight budget.",
            height=100,
        )

    hypothesis_to_validate = st.text_input(
        "Hypothesis to Validate",
        placeholder="e.g., Students are willing to rent dresses from peers rather than buy new ones.",
    )

    # ----------------------------------------------------------------------------------
    # OPTIONAL: PERSONA BUILDER
    # ----------------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("👤 Persona Builder (Optional)")
    st.markdown("Need help making your customer segment realistic? Generate a backstory first.")

    if st.button("✨ Generate Persona Backstory"):
        if not api_key:
            st.error("Please enter your LLM API Key in the sidebar first.")
        elif not problem_statement or not customer_segment:
            st.error("Please fill in Problem Statement and Customer Segment first.")
        else:
            try:
                client = build_client(api_key=api_key, base_url=base_url)
                with st.spinner("Dreaming up a customer..."):
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": build_persona_prompt(customer_segment, problem_statement)}],
                        reasoning_effort="low",
                    )
                st.session_state.generated_persona = response.choices[0].message.content.replace("$", r"\$")
            except Exception as exc:
                st.error(f"Error: {exc}")

    # Track whether the user explicitly opts into the generated persona (logic preserved; original code defaulted to using persona if generated)
    use_generated_persona = False

    if st.session_state.generated_persona:
        st.success("**Generated Persona:**\n\n" + st.session_state.generated_persona)

        persona_choice = st.radio(
            "Which persona context do you want to use?",
            ["Use Generated Persona", "Use Raw User Inputs Only"],
            index=0,
            horizontal=True,
        )
        if persona_choice == "Use Generated Persona":
            use_generated_persona = True

    st.markdown("---")

    # ----------------------------------------------------------------------------------
    # START INTERVIEW ACTION
    # ----------------------------------------------------------------------------------
    if st.button("🚀 Start Interview", type="primary"):
        if not api_key:
            st.error("Please enter your LLM API Key in the sidebar.")
        elif not problem_statement or not customer_segment or not hypothesis_to_validate:
            st.error("Please fill in all the input fields above.")
        else:
            persona_context = (
                st.session_state.generated_persona
                if (st.session_state.generated_persona and use_generated_persona)
                else f"A member of this segment: {customer_segment}"
            )


            system_prompt = build_system_prompt(
                problem_statement=problem_statement,
                hypothesis_to_validate=hypothesis_to_validate,
                persona_context=persona_context,
            )

            st.session_state.messages = [{"role": "system", "content": system_prompt}]
            st.session_state.interview_active = True
            st.rerun()

# ======================================================================================
# SECTION 2: INTERVIEW INTERFACE
# ======================================================================================
if st.session_state.interview_active and not st.session_state.analysis_done:
    st.subheader("💬 Interview in Progress")

    avatar_map = {
        "user": "🧑‍🎓",
        "assistant": "👤",
    }

    # Display chat history (excluding system prompt)
    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"], avatar=avatar_map.get(message["role"], "💬")):
            st.markdown(message["content"])

    # User input -> assistant response (streamed)
    user_question = st.chat_input("Start asking your question to the customer here...")
    if user_question:
        st.chat_message("user", avatar=avatar_map["user"]).write(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        client = build_client(api_key=api_key, base_url=base_url)
        with st.chat_message("assistant", avatar=avatar_map["assistant"]):
            stream = client.chat.completions.create(
                model=model_name,
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                reasoning_effort="low",
                stream=True,
            )
            assistant_reply = st.write_stream(escape_dollars(stream))

        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

    st.markdown("---")
    if len(st.session_state.messages) > 2:  # Only show if conversation has started
        if st.button("🛑 End & Analyze Interview", type="tertiary"):
            st.session_state.analysis_done = True
            st.rerun()

# ======================================================================================
# SECTION 3: ANALYSIS & FEEDBACK
# ======================================================================================
if st.session_state.analysis_done:
    st.subheader("📝 AI Professor's Feedback")

    transcript = transcript_from_messages(st.session_state.messages)
    coach_prompt = build_coach_prompt(transcript)

    if api_key and not st.session_state.feedback_text:
        with st.spinner("Analyzing your interview technique..."):
            try:
                client = build_client(api_key=api_key, base_url=base_url)
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": coach_prompt}],
                    reasoning_effort="low",
                    stream=True,
                )
                feedback_markdown = st.write_stream(escape_dollars(stream))
                st.session_state.feedback_text = feedback_markdown

                if st.session_state.feedback_text:
                    st.download_button(
                        "Download Feedback",
                        data=st.session_state.feedback_text,
                        file_name="interview_feedback.txt",
                        icon=":material/download:",
                    )
            except Exception as exc:
                st.error(f"Error analyzing: {exc}")

    if st.button("Start New Simulation", icon=":material/playlist_add:", type="primary"):
        reset_simulation()
