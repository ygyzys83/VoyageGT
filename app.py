import streamlit as st
import os
import re
from agent import voyage_agent, MODEL_PROVIDER
from langchain_core.messages import HumanMessage, AIMessage
from tools import send_itinerary_email

st.set_page_config(
    page_title="VoyageGT",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        section[data-testid="stSidebar"] { width: 360px !important; min-width: 360px !important; }
        .main .block-container { padding-left: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🌍 VoyageGT")
st.markdown("### Your Intelligent Agentic Travel Planner")
st.caption("Plan any trip in the world with real-time research and smart reasoning")

# Sidebar
with st.sidebar:
    st.header("📋 Trip Basics")
    num_travelers = st.number_input("Number of Travelers", min_value=1, value=2, step=1)
    travel_dates = st.text_input("Travel Dates", placeholder="e.g., June 15-30 2026 or September-October")

    col1, col2 = st.columns(2)
    with col1:
        from_city = st.text_input("From City", placeholder="Amsterdam")
    with col2:
        to_city = st.text_input("To City", placeholder="Berlin")

    additional_stops = st.text_input("Additional Stops (optional)", placeholder="e.g., Paris, Munich")

    budget_level = st.selectbox("Budget Level", options=["Budget", "Mid-range", "Comfort", "Luxury"], index=1)
    travel_style = st.text_area("Travel Style / Preferences", placeholder="e.g., love history and food...", height=120)

    st.divider()
    if st.button("🚀 Start New Trip with These Details", type="primary"):
        summary = f"I am planning a trip with these details:\n- Travelers: {num_travelers}\n- Dates: {travel_dates}\n- From: {from_city}\n- To: {to_city}\n- Additional stops: {additional_stops if additional_stops else 'None'}\n- Budget: {budget_level}\n- Style: {travel_style if travel_style else 'Not specified'}"
        st.session_state.messages = [{"role": "user", "content": summary.strip()}]
        st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Powered by Qwen3.6 + LangGraph + Tavily")

# Initialize observability
if "thinking_trace" not in st.session_state:
    st.session_state.thinking_trace = []

# Chat area
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Describe your trip idea or ask VoyageGT anything..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("VoyageGT is planning your trip..."):
            history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                       for m in st.session_state.messages]

            response = voyage_agent.invoke({"messages": history})

            # Extract final message
            final_message = None
            for msg in reversed(response.get("messages", [])):
                content = getattr(msg, 'content', None)
                if isinstance(content, str) and content.strip():
                    content = re.sub(r'-{10,}|={10,}|_{10,}', '', content)
                    content = re.sub(r'\n{4,}', '\n\n\n', content)
                    content = re.sub(r'[A-Za-z0-9+/]{80,}', '', content)
                    final_message = content.strip()
                    break

            if final_message:
                st.markdown(final_message)
            else:
                st.error("Sorry, I couldn't generate a response. Please try again.")

            # === Improved Observability: Agent Thinking Trace ===
            trace = []
            for msg in response.get("messages", []):
                content_str = str(getattr(msg, 'content', msg))

                if "search_web" in content_str.lower() or "tavily" in content_str.lower():
                    trace.append("🔍 **Tavily search_web called** (Grounded Research)")
                elif hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        trace.append(
                            f"🛠️ **Tool call:** `{tc.get('name', 'unknown')}` — args: {str(tc.get('args', {}))[:100]}")
                elif "tool_call_id" in content_str:
                    trace.append(f"✅ **Tool result received:** {content_str[:120]}...")
                elif len(content_str) > 10:
                    trace.append(content_str[:250] + "..." if len(content_str) > 250 else content_str)

            st.session_state.thinking_trace.append({
                "user_prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
                "trace": trace[-8:]  # keep last 8 steps for clarity
            })

    if final_message:
        st.session_state.messages.append({"role": "assistant", "content": final_message})

# ================== AGENT THINKING / OBSERVABILITY SECTION ==================
if st.session_state.thinking_trace:
    with st.expander("🔍 Agent Thinking & Decision Log (Observability)", expanded=False):
        for i, entry in enumerate(reversed(st.session_state.thinking_trace[-6:])):  # show last 6 interactions
            st.subheader(f"Interaction {len(st.session_state.thinking_trace) - i}")
            st.caption(f"**User:** {entry['user_prompt']}")
            for step in entry["trace"]:
                st.markdown(f"• {step}")
            st.divider()

# Email button
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📧 Send This Itinerary to Email", type="primary", use_container_width=True):
            with st.spinner("Sending email via Gmail..."):
                last_response = st.session_state.messages[-1]["content"]
                if last_response and len(last_response.strip()) > 100:
                    result = send_itinerary_email.invoke({
                        "recipient_email": os.getenv("GMAIL_USER"),
                        "subject": "VoyageGT Trip Itinerary - Your Personalized Plan",
                        "itinerary_content": last_response
                    })
                    if "✅" in result:
                        st.success(result)
                    else:
                        st.error(result)
                else:
                    st.error("No valid itinerary found to send.")