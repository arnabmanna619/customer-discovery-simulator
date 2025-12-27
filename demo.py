import streamlit as st
from openai import OpenAI

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Customer Discovery Simulator", page_icon="🕵️", layout="wide")

# --- STATE MANAGEMENT ---
# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "interview_active" not in st.session_state:
    st.session_state.interview_active = False
if "generated_persona" not in st.session_state:
    st.session_state.generated_persona = ""
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. Provider Selection
    llm_provider = st.selectbox(
        "Select LLM Provider", 
        ["OpenAI", "Gemini", "Gemini (Test)"]
    )

    # 2. Logic for API Keys and Base URLs
    if llm_provider == "OpenAI":
        user_api_key = st.text_input("OpenAI API Key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/docs/quickstart)"
        api_key = user_api_key
        base_url = None # Default OpenAI URL
        model_name = "gpt-5.1" 
        
    elif llm_provider == "Gemini":
        user_api_key = st.text_input("Gemini API Key", type="password")
        "[Get Gemini API key](https://ai.google.dev/gemini-api/docs/api-key)"
        api_key = user_api_key
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        model_name = "gemini-3-flash-preview" 
        
    else: # Gemini (Test)
        st.success("Using a Free Test Key. This has limits.")
        api_key = st.secrets.get("GEMINI_TEST_API_KEY", "")
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        model_name = "gemini-3-flash-preview"

    st.info(
        "**Instructions:**\n"
        "1. Fill in your hypothesis details.\n"
        "2. (Optional) Generate a specific persona.\n"
        "3. Click 'Start Interview' to chat with the AI.\n"
        "4. Click 'End & Analyze' to get graded."
    )
    
    if st.button("New Simulation", icon=":material/playlist_add:", type="secondary"):
        st.session_state.messages = []
        st.session_state.interview_active = False
        st.session_state.generated_persona = ""
        st.session_state.analysis_done = False
        st.rerun()

# --- MAIN PAGE: HEADER ---
st.title("🕵️ Customer Discovery Simulator")
st.markdown("Practice your interview skills on an AI persona before talking to real humans.")

def escape_dollars(stream):
    for event in stream:
        delta = event.choices[0].delta
        text = getattr(delta, "content", None)
        if text:
            yield text.replace("$", r"\$")

# --- SECTION 1: INPUTS ---
# Only show inputs if the interview hasn't started yet
if not st.session_state.interview_active:
    
    col1, col2 = st.columns(2)
    
    with col1:
        problem = st.text_area(
            "Problem Statement", 
            placeholder="e.g., College students spend too much money buying formal wear for one-time events.",
            height=100
        )
    with col2:
        segment = st.text_area(
            "Customer Segment", 
            placeholder="e.g., A Sophomore at UPenn who attends 3-4 formals a year but is on a tight budget.",
            height=100
        )
        
    hypothesis = st.text_input(
        "Hypothesis to Validate", 
        placeholder="e.g., Students are willing to rent dresses from peers rather than buy new ones."
    )

    # --- OPTIONAL: PERSONA BUILDER ---
    st.markdown("---")
    st.subheader("👤 Persona Builder (Optional)")
    st.markdown("Need help making your customer segment realistic? Generate a backstory first.")
    
    if st.button("✨ Generate Persona Backstory"):
        if not api_key:
            st.error("Please enter your LLM API Key in the sidebar first.")
        elif not problem or not segment:
            st.error("Please fill in Problem Statement and Customer Segment first.")
        else:
            try:
                client = OpenAI(api_key=api_key, base_url=base_url)
                with st.spinner("Dreaming up a customer..."):
                    # Prompt to generate a backstory
                    persona_prompt = (
                        f"I am an entrepreneur. My customer segment is: {segment}. "
                        f"The problem they face is: {problem}. "
                        "Create a brief, specific 3-line persona description for a mock interview. "
                        "Include a specific name, major/job, and a hidden 'ground truth' about how they currently solve this problem."
                    )
                    
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": persona_prompt}],
                        reasoning_effort="low"
                    )
                    st.session_state.generated_persona = response.choices[0].message.content.replace("$", r"\$")
            except Exception as e:
                st.error(f"Error: {e}")

    # VARIABLE TO TRACK USER CHOICE (Default to False)
    use_generated_persona = False

    # Display generated persona if it exists AND offer choice
    if st.session_state.generated_persona:
        st.success("**Generated Persona:**\n\n" + st.session_state.generated_persona)
        
        # User decides: Use this fancy persona OR just the raw inputs?
        persona_choice = st.radio(
            "Which persona context do you want to use?",
            ["Use the Generated Persona Details", "Ignore and use my Raw Inputs only"],
            index=0 # Default to using the generated one since they asked for it
        )
        
        if persona_choice == "Use the Generated Persona Details":
            use_generated_persona = True

    st.markdown("---")

    # --- START INTERVIEW ACTION ---
    if st.button("🚀 Start Interview", type="primary"):
        if not api_key:
            st.error("Please enter your LLM API Key in the sidebar.")
        elif not problem or not segment or not hypothesis:
            st.error("Please fill in all the input fields above.")
        else:
            # Construct the System Prompt
            # If they generated a persona, we use that. If not, we use the raw inputs.
            persona_context = st.session_state.generated_persona if st.session_state.generated_persona else f"A member of this segment: {segment}"
            
            system_prompt = f"""
            I want you to act as a specific persona for a mock customer discovery interview. Do not break character until I say "STOP INTERVIEW."
            
            **My Business Context:**
            I am investigating a problem related to: {problem}.
            My Hypothesis is: {hypothesis}.
            
            **Your Persona:**
            {persona_context}
            
            **Rules for your Roleplay:**
            1. Be realistic. Real people are busy and sometimes indifferent.
            2. If I ask a "Yes/No" question, give a short answer.
            3. If I ask a "Leading Question" (e.g., "Wouldn't you love an app that...?"), react with skepticism or polite disinterest.
            4. If I ask a great "Open-Ended Question" (e.g., "Tell me about the last time you...?"), open up and share details.
            5. DO NOT offer solutions. Only talk about your life, problems, and current behaviors.
            """
            
            # Initialize chat history
            st.session_state.messages = [{"role": "system", "content": system_prompt}]
            st.session_state.interview_active = True
            st.rerun()

# --- SECTION 2: INTERVIEW INTERFACE ---
if st.session_state.interview_active and not st.session_state.analysis_done:
    st.subheader("💬 Interview in Progress")
    
    # Display Chat History (excluding system prompt)
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # User Input
    if user_input := st.chat_input("Start asking your question to the customer here..."):
        # 1. Display User Message
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 2. Stream AI Response
        client = OpenAI(api_key=api_key, base_url=base_url)
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                reasoning_effort="low",
                stream=True,
            )
            response = st.write_stream(escape_dollars(stream))
        
        # 3. Save AI Response
        st.session_state.messages.append({"role": "assistant", "content": response})

    # End Interview Button
    st.markdown("---")
    if len(st.session_state.messages) > 2: # Only show if conversation has started
        if st.button("🛑 End & Analyze Interview", type="tertiary"):
            st.session_state.analysis_done = True
            st.rerun()

# --- SECTION 3: ANALYSIS & FEEDBACK ---
if "feedback_text" not in st.session_state:
    st.session_state.feedback_text = ""

if st.session_state.analysis_done:
    st.subheader("📝 AI Professor's Feedback")
    
    # Combine the chat history into a string for the LLM to read
    transcript = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages if m['role'] != 'system'])
    
    coach_prompt = f"""
    You are an Expert Entrepreneurship Professor (like the one teaching MGMT 801 at Wharton). 
    Please analyze the transcript of the student's mock interview below.
    
    TRANSCRIPT:
    {transcript}
    
    Provide feedback in markdown format:
    1. **Question Quality:** Did the student ask "Leading Questions" (bad) or "Open-Ended Questions" (good)? Quote a specific mistake if present.
    2. **Hypothesis Validation:** Based ONLY on the persona's answers, was the hypothesis validated or invalidated?
    3. **Missing Info:** What is one critical thing they failed to ask?
    4. **Grading:** Give a score out of 10 for unbiased interviewing technique.
    """
    
    if api_key and not st.session_state.feedback_text:
        with st.spinner("Analyzing your interview technique..."):
            try:
                client = OpenAI(api_key=api_key, base_url=base_url)
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": coach_prompt}],
                    reasoning_effort="low",
                    stream=True
                )
                feedback = st.write_stream(escape_dollars(stream))
                st.session_state.feedback_text = feedback

                # Display cached feedback for download
                if st.session_state.feedback_text:
                    st.download_button("Download Feedback", data=st.session_state.feedback_text, file_name="interview_feedback.txt", icon=":material/download:")
                
            except Exception as e:
                st.error(f"Error analyzing: {e}")
    
    if st.button("Start New Simulation", icon=":material/playlist_add:"):
        st.session_state.messages = []
        st.session_state.interview_active = False
        st.session_state.generated_persona = ""
        st.session_state.analysis_done = False
        st.rerun()
