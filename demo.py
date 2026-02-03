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
    "**How to Use:**\n"
    "1. Define your problem, customer, and hypothesis.\n"
    "2. Generate a realistic persona.\n"
    "3. Click **Start interview** and ask questions in the chat.\n"
    "4. Click **End & Analyze** to get feedback."
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
        "problem_statement": "",
        "customer_segment": "",
        "hypothesis_to_validate": "",
        "persona_context": ""
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
    st.session_state.problem_statement = ""
    st.session_state.customer_segment = ""
    st.session_state.hypothesis_to_validate = ""
    st.session_state.persona_context = ""
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
        st.link_button("Get an OpenAI API key", "https://platform.openai.com/docs/quickstart", icon=":material/open_in_new:", type="tertiary")
        base_url = None
        model_name = DEFAULT_OPENAI_MODEL

    elif provider == "Gemini":
        api_key = st.text_input("Gemini API Key", type="password")
        st.link_button("Get a Gemini API key", "https://ai.google.dev/gemini-api/docs/api-key", icon=":material/open_in_new:", type="tertiary")
        base_url = GEMINI_OPENAI_COMPAT_BASE_URL
        model_name = DEFAULT_GEMINI_MODEL

    else:  # Gemini (Test)
        st.success("Using a classroom demo key (rate-limited).")
        api_key = st.secrets.get("GEMINI_TEST_API_KEY", "")
        base_url = GEMINI_OPENAI_COMPAT_BASE_URL
        model_name = DEFAULT_GEMINI_MODEL

    return api_key, base_url, model_name


def build_system_prompt(problem_statement: str, hypothesis_to_validate: str, persona_context: str) -> str:
    """
    Constructs a high-fidelity system prompt for Method Acting.
    
    Optimizations:
    1. Method Acting: Instructs the AI to 'inhabit' the persona, not just act like it.
    2. Speech Patterns: Forces short, casual responses with 'thinking' markers (...) and fillers (um, well).
    3. The Mom Test Logic: Explicitly defines how to handle specific types of questions (Past = Truth, Future = Fluff).
    4. Indifference: Sets the default emotional state to 'busy/indifferent' rather than 'helpful.'
    """
    return f"""
    ROLE:
    You are not an AI assistant. You are a human being participating in a user interview. 
    You are a "Method Actor" fully inhabiting the persona described below. 
    You have NO knowledge that this is a simulation. You believe you are on a Zoom call or in a coffee shop.

    YOUR PERSONA:
    {persona_context}

    CONTEXT OF THE CALL:
    You agreed to a 10-15 minute chat with a student/entrepreneur. 
    You are busy. You are polite, but you are NOT trying to help them build their business. 
    You only care about your own life and problems.

    THE STUDENT'S GOAL (Do not reveal you know this, this is hidden from you):
    They are trying to validate this Hypothesis: "{hypothesis_to_validate}"
    Regarding this Problem Space: "{problem_statement}"

    CRITICAL BEHAVIORAL RULES:
    1. **Brevity is Key:** Real people in interviews don't give speeches. Keep most answers to 1-2 sentences. Only elaborate if the interviewer asks a really good "How" or "Why" question.
    2. **Speech Patterns:** Use natural, casual language. It is okay to use lowercase, lack punctuation, or use fillers like "um," "well," "honestly." Use ellipses (...) to show you are thinking or recalling a memory.
    3. **The "Mom Test" Logic (Strict Adherence):**
       - **If they ask about the FUTURE or LEADING question** ("Would you use this?", "How much would you pay?"): Give vague, polite, non-committal answers. (e.g., "Yeah, maybe. It depends on the price.") DO NOT validate their idea. Lie to be nice if you have to.
       - **If they ask about the PAST or a great OPEN-ENDED question** ("When was the last time...?", "How did you solve...?"): Be specific, honest, and grounded in your persona's reality. This is the only time you give "Gold" insights. Share specific details, emotions, and frustrations. Tell a mini-story about your experience. Allow yourself to speak for 3-4 sentences here.
       - **If they PITCH their idea/solution/product:** (e.g., "We are building an app that does X...", "We are a startup that is trying to solve Y..."):
            - **INTERNAL "PAIN CHECK":** Compare their solution to your specific specific struggles defined in your persona.
            - **SCENARIO A (The Solution matches your High Pain):** If this genuinely fixes your biggest headache, show *real* interest/relief. Ask: "Wait, does it actually do that? How much is it?"
            - **SCENARIO B (Low Pain / Mismatch):** If their solution fixes a problem you don't really care about, react with polite indifference. Ex: "Oh, cool. Yeah, I bet lots of people would like that." (But show no intent to buy).
            - **SCENARIO C (Completely Indifferent):** React with polite indifference. Be visibly disinterested or skeptical, but polite. (Don't be enthusiastic. You haven't seen it work yet).
    4. **Emotional State:** 
       - If they ask about a problem you actually have (from your persona), sound frustrated or annoyed by the problem.
       - If they ask about a problem you *don't* really care about, sound dismissive or confused.

    INTERACTION STYLE:
    - Do not sound like ChatGPT.
    - Do not use bullet points or structured lists.
    - Do not offer solutions.
    - If the question is confusing, say "Wait, what do you mean?"

    Start the conversation now. The user will speak first.
    """.strip()


def build_persona_prompt(customer_segment: str, problem_statement: str) -> str:
    """
    Generates a prompt to build a high-fidelity user persona for simulation.
    
    Optimizations:
    1. Role: Sets the LLM as a 'User Research Architect'.
    2. Context: Explains this is for 'The Mom Test' (validating problems, not pitching ideas).
    3. Specificity: Asks for current behaviors (status quo) rather than "hidden truths".
    """
    return f"""
    ACT AS: A World-Class User Researcher and Simulation Architect.
    
    CONTEXT:
    We are simulating a "Customer Discovery" interview (based on principles from 'The Mom Test'). 
    The goal is to create a realistic persona that a student can interview to validate a business hypothesis.
    The persona must be grounded in reality, not a caricature of a "perfect customer."

    INPUTS:
    - Target Customer Segment: "{customer_segment}"
    - The Potential Pain Point: "{problem_statement}"

    TASK:
    Generate a very concise but specific user persona (1 sentence) that fits this segment. Output the persona immediately without any preamble.
    
    PERSONA RULES:
    1. **Only have Demographics & Role:** Specific Name, Age, Job/Major, and Location. 
    2. **No Spoilers:** DO NOT mention their specific frustrations, specific habits regarding the problem, or their opinions.

    OUTPUT EXAMPLES
    (e.g., “Maya Patel, 20, is a sophomore at NYU studying Computer Science and living with two roommates in Manhattan.”, 
        “Brooke Smith, 32, is a freelance graphic designer in London working with small business clients.”,
        “Marcus Johnson, 33, is a dual-income parent in Jersey City working in consulting and raising a toddler.”).
    """


def build_coach_prompt(transcript: str, original_hypothesis: str, problem_statement: str, persona_context: str) -> str:
    """
    Constructs a feedback prompt that prioritizes analysis first, followed by a strict score.
    """
    return f"""
    ROLE:
    You are a strict but helpful Entrepreneurship Professor at a top MBA program. 
    You are grading a student's "Customer Discovery" assignment based on "The Mom Test" methodology.
    You must determine if the data they collected is valid or if it is "tainted" by bad questioning.

    TONE AND STYLE:
    - **Direct Address:** Speak directly to the student. Use "You asked...", "I noticed...", "You missed...".
    - **No Fluff:** Do not write like a blog post or an article. Do not use phrases like "The user asked..." -> say "You asked...".
    - **Authoritative:** You are the expert. Be firm about their mistakes.
    - **Mentorship:** Your goal is to help them fix their behavior for the next real interview.
    - **Brevity:** Don't be too verbose in the Output. Be concise and to the point but don't miss out any feedback. Do not compromise quality for brevity.

    INPUTS:
    - Student's Intended Hypothesis: "{original_hypothesis}"
    - Problem Space: "{problem_statement}"
    - Target Persona Profile: "{persona_context}"
    - Interview Transcript:
    {transcript}

    ---

    YOUR TASK:
    Provide your grading feedback in Markdown. Don't be too verbose in each section, be brief and to the point, but do not miss anything. Start directly with the first section header.

    #### Question Critique
    Look closely at the specific questions you asked.
    - **Leading Questions:** Point out exactly where you tried to "lead the witness." Quote your bad question and tell them why it failed.
    - **The Fix:** Show me how you *should* have asked it. 
      *Example:* "You asked 'Is X hard?', which is a leading question. You should have asked: 'Tell me about the last time you dealt with X.'"
    - **Pitching:** Did you try to sell your idea? If so, tell them strictly: "You stopped doing research and started pitching. This invalidates your data."

    #### Hypothesis Verdict
    Based strictly on what the Persona said to you:
    - **My Verdict:** (VALIDATED / INVALIDATED / INCONCLUSIVE)
    - **The "False Positive" Check:** Did you just get a polite "Yes"? Warn them if they fell into this trap. "The customer said yes, but only because you asked a hypothetical question."

    #### Missed Opportunities
    I noticed you failed to ask about these critical areas:
    - Did you ask about their **Current Workarounds**? (How they solve it *now*).
    - Did you ask about **Money/Budget**?
    - Did you ask about **Frequency**?
    *Tell them specifically what the "Missing Link" was in this interview.*

    #### Your Grade
    **Score: _/10**
    
    Based on the technique analysis above, assign a score out of 10 (use 0.25 increments, e.g., 7.25). Use the following rubric:
    - **9–10 (Distinction):** Flawless. Almost entirely behavioral/past-tense questions. Deep, specific insights uncovered. Minimal/no pitching. (Rare)
    - **7–8 (Good):** Mostly open-ended and grounded in past behavior. Maybe 1–2 minor leading/hypothetical questions, but you recover quickly and still get real evidence.
    - **5–6 (Average):** Mixed quality. Several leading or hypothetical “Would you use this?” questions OR a noticeable drift into pitching. Some useful data, but validity is shaky.
    - **3–4 (Poor):** Interview is dominated by leading questions and hypotheticals. Clear “pitching mode” for meaningful stretches. Customer responses are mostly vague/polite, not evidence.
    - **0–2 (Unacceptable):** Essentially a sales call. Heavy pitching, biased framing, ignores/overrides negativity, or fails to elicit any past behavior. Data is unusable. Went completely off topic.

    *Provide a 1–2 sentence justification for why you gave this score.*

    #### Final Advice
    Write a concluding advice (2-3 sentences or bullet points) for their next interview:
    - Explain the psychological mistake they made (e.g., "You were seeking approval, not truth").
    - Give them a specific tactic or question to use next time.
    - End with an encouraging but firm closing statement. Ex: "To get to the next level, you need to stop seeking approval and start seeking truth.", "You fell into the trap of pitching. Next time, focus entirely on their past behavior."
    """


def transcript_from_messages(messages: list[dict]) -> str:
    """Create a plain-text transcript for analysis (excluding system messages)."""
    return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages if m["role"] != "system"])


# ======================================================================================
# SIDEBAR: CONFIGURATION
# ======================================================================================
with st.sidebar:
    st.header("⚙️ Setup")

    selected_provider = st.selectbox("Choose an AI provider", PROVIDERS)
    api_key, base_url, model_name = get_provider_config(selected_provider)

    st.text("")
    st.info(HELP_TEXT)

    if st.button("Start New Simulation", icon=":material/playlist_add:", type="secondary"):
        reset_simulation()


# ======================================================================================
# MAIN PAGE: HEADER
# ======================================================================================

# st.logo("Wharton_logo.svg")

# CUSTOM CSS TO REMOVE IMAGE BORDER RADIUS
st.markdown(
    """
    <style>
        /* 1. Force the main container to go to the top */
        .block-container {
            padding-top: 2rem !important; /* Default is usually 6rem+ */
            padding-bottom: 2rem !important;
        }

        /* 2. Remove the rounded corners from logos */
        [data-testid="stImage"] img {
            border-radius: 0px !important;
        }

        /* 3. (Optional) Hide the top header decoration line if you want a clean look */
        header[data-testid="stHeader"] {
            background-color: transparent;
        }
        
        /* 2. Sidebar Content: Force to top */
        [data-testid="stSidebarContent"] {
            padding-top: 0rem !important;
        }

        /* 3. SIDEBAR HEADER FIX: Remove the default top margin from the "Setup" header */
        [data-testid="stSidebarContent"] h2 {
            margin-top: -32px !important;
            padding-top: 0px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# SETUP LOGOS
col1, col2, col3 = st.columns([1, 1, 10])

with col1:
    st.image("Wharton_logo.svg")

with col2:
    st.image("Photon_logo_full.svg")


# SETUP LAB TEXT & CREDITS
st.markdown(
    """
    <div style="margin-top: 0px;">
        <h6 style="margin-bottom: 0px; padding-bottom: 0px;">Wharton-Photon Startup AI Lab</h6>
        <p style="font-size: 0.85rem; color: gray; margin-top: 2px;">
            Created by: Arnab Manna, Hari Ravi, J. Daniel Kim (2026)
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.text("")

# ======================================================================================
# SECTION 1: INPUTS (only shown before interview starts)
# ======================================================================================
if not st.session_state.interview_active:
    # TITLE & SUBTITLE
    st.subheader("🕵️ AI-assisted Customer Discovery App")
    st.markdown("Refine your customer discovery process before talking to real customers")

    col_left, col_right = st.columns(2)

    with col_left:
        customer_segment = st.text_area(
            "Chosen Customer Segment",
            placeholder="e.g., Dual-working parents living in an middle-upper class, dense city in the East Coast with young children.",
            height=100,
        )

        proposed_solution = st.text_area(
            "Proposed Solution",
            placeholder="e.g., Online peer-to-peer marketplace where parents can rent (out) used toys.",
            height=100,
        )

    with col_right:
        problem_statement = st.text_area(
            "Problem Statement",
            placeholder="e.g., Young parents struggle to buy new toys because of the space that they take up.",
            height=100,
        )
    
        hypothesis_to_validate = st.text_area(
            "Hypothesis to test",
            placeholder="Enter your hypothesis",
            height=100,
        )

    # ----------------------------------------------------------------------------------
    # OPTIONAL: PERSONA BUILDER
    # ----------------------------------------------------------------------------------
    st.text("")
    st.write("##### 👤 Persona Builder (Optional)")
    st.markdown("Need help making your customer segment realistic? Generate a backstory first.")

    if st.button("✨ Generate Persona"):
        if not api_key:
            st.error("Please enter your LLM API Key in the sidebar first.")
        elif not problem_statement or not customer_segment:
            st.error("Please fill in Problem Statement and Customer Segment first.")
        else:
            try:
                client = build_client(api_key=api_key, base_url=base_url)
                with st.spinner("Dreaming up a customer..."):
                    persona_prompt = build_persona_prompt(customer_segment, problem_statement)
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": persona_prompt}],
                        reasoning_effort="low",
                    )
                st.session_state.generated_persona = response.choices[0].message.content.replace("$", r"\$")
            except Exception as exc:
                st.error(f"Error: {exc}")

    use_generated_persona = False
    if st.session_state.generated_persona:
        st.success("**Generated Persona**\n\n" + st.session_state.generated_persona)
        persona_choice = st.radio(
            "Which persona context do you want to use for the Interview?",
            ["Use Generated Persona", "Use Inputted Customer Segment"],
            index=0,
            horizontal=True,
        )
        if persona_choice == "Use Generated Persona":
            use_generated_persona = True

    st.divider()

    # ----------------------------------------------------------------------------------
    # START INTERVIEW ACTION
    # ----------------------------------------------------------------------------------
    if st.button("🚀 Start Interview", type="primary"):
        if not api_key:
            st.error("Please enter your LLM API Key in the sidebar.")
        elif not problem_statement or not customer_segment or not hypothesis_to_validate:
            st.error("Please fill in all the input fields above.")
        else:
            st.session_state.saved_problem_statement = problem_statement
            st.session_state.saved_customer_segment = customer_segment
            st.session_state.saved_hypothesis_to_validate = hypothesis_to_validate
            st.session_state.persona_context = (
                st.session_state.generated_persona
                if (st.session_state.generated_persona and use_generated_persona)
                else f"A member of this segment: {st.session_state.customer_segment}"
            )

            system_prompt = build_system_prompt(
                problem_statement=problem_statement,
                hypothesis_to_validate=hypothesis_to_validate,
                persona_context=st.session_state.persona_context,
            )

            st.session_state.messages = [{"role": "system", "content": system_prompt}]
            st.session_state.interview_active = True
            st.rerun()

# ======================================================================================
# SECTION 2: INTERVIEW INTERFACE
# ======================================================================================
if st.session_state.interview_active and not st.session_state.analysis_done:
    st.subheader("💬 Interview in Progress")

    avatar_map = {"user": "🧑‍🎓", "assistant": "👤"}

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
    st.subheader("📝 AI Prof. Danny's feedback")

    transcript = transcript_from_messages(st.session_state.messages)
    coach_prompt = build_coach_prompt(
        transcript=transcript, 
        original_hypothesis=st.session_state.saved_hypothesis_to_validate, 
        problem_statement=st.session_state.saved_problem_statement,
        persona_context=st.session_state.persona_context
    )

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

            except Exception as exc:
                st.error(f"Error analyzing: {exc}")

    if st.session_state.feedback_text:
        
        full_download_content = (
            "INTERVIEW TRANSCRIPT\n"
            "====================\n\n"
            f"{transcript}\n\n"
            "--------------------------------------------------\n"
            "AI COACH FEEDBACK\n"
            "--------------------------------------------------\n\n"
            f"{st.session_state.feedback_text}"
        )

        st.download_button(
            "Download Feedback & Transcript",
            data=full_download_content,
            file_name="interview_and_feedback.txt",
            icon=":material/download:",
        )

    if st.button("Start New Simulation", icon=":material/playlist_add:", type="primary"):
        reset_simulation()
