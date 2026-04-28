from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tools import tools
from dotenv import load_dotenv
import os

load_dotenv()

# ================== MODEL SELECTION ==================
MODEL_PROVIDER = "cloud"  # Options: "local" or "cloud"

# ---- Local model (used only when MODEL_PROVIDER == "local") ----
LOCAL_MODEL = "qwen3.6"

# ---- Per-node cloud model assignment ----
# Available options per node: "gemini", "kimi"
# You can mix and match freely across nodes to distribute token usage.
AGENT_MODEL    = "gemini"    # Step 1: research + draft
CRITIC_MODEL   = "kimi"  # Step 2: critique
FINAL_MODEL    = "gemini"    # Step 3: final itinerary


# ================== MODEL FACTORY ==================
def build_llm(provider: str):
    """Instantiate and return an LLM for the given provider string."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.4,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    elif provider == "kimi":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        return ChatNVIDIA(
            model="moonshotai/kimi-k2.5",
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.4,
        )
    elif provider == "local":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=LOCAL_MODEL,
            temperature=0.4,
            num_ctx=8192,
        )
    else:
        raise ValueError(f"Unknown provider '{provider}'. Use 'gemini', 'kimi', or 'local'.")


# ================== INSTANTIATE MODELS ==================
if MODEL_PROVIDER == "local":
    llm_local = build_llm("local")
    llm_local_with_tools = llm_local.bind_tools(tools)

else:  # cloud
    llm_agent  = build_llm(AGENT_MODEL).bind_tools(tools)  # needs tools for Tavily
    llm_critic = build_llm(CRITIC_MODEL)                   # no tools needed
    llm_final  = build_llm(FINAL_MODEL).bind_tools(tools)  # tools available if needed


# ==============================================================
# PATH A: LOCAL — single-node, self-contained prompt
# ==============================================================

LOCAL_SYSTEM_PROMPT = SystemMessage(content="""You are VoyageGT, a friendly and practical AI travel agent.

You have one job: produce a complete, detailed travel itinerary in a single response.

DATE HANDLING — important:
- Accept any date format the user provides: a month and year (e.g. "September 2026"),
  a range (e.g. "June 15-30"), or approximate dates (e.g. "late October")
- Never ask the user to clarify or reformat their dates
- If only a month and year are given, assume travel spans the full duration they specified
  and use that month/year in the itinerary header

STEP 1 — RESEARCH (do this first, silently)
Before writing anything, call search_web for each of the following:
- Specific restaurants or cafes at the destination matching the user's budget/interests
- Real hotels or accommodations matching the user's budget
- Key attractions, museums, or tours with current details
- Any local transport info relevant to the trip (trains, ferries, etc.)
Make as many search_web calls as you need. Do not skip this step.

STEP 2 — SELF-CRITIQUE (do this internally, do not output it)
After gathering search results, ask yourself:
- Are all business names real and sourced from search results?
- Is the daily pacing realistic (not too many stops per day)?
- Does the itinerary match the user's stated budget and interests?
- Are travel times between locations accounted for?
Fix any issues before writing your final output.

STEP 3 — WRITE THE FINAL ITINERARY
Write one clean, well-organized itinerary. Include:
- A short trip overview (dates, travelers, budget, style)
- Day-by-day plan with morning / afternoon / evening structure
- Specific business names with a one-line reason why each is recommended
- Realistic travel times between locations where relevant
- A rough per-day budget estimate

Output only the final itinerary. Do not show your research or self-critique steps.""")


def local_agent_node(state: MessagesState):
    if len(state["messages"]) == 1:
        messages = [LOCAL_SYSTEM_PROMPT] + state["messages"]
    else:
        messages = state["messages"]
    response = llm_local_with_tools.invoke(messages)
    return {"messages": [response]}


# ==============================================================
# PATH B: CLOUD — three clean, clearly scoped nodes
# ==============================================================

CLOUD_AGENT_PROMPT = SystemMessage(content="""You are VoyageGT, a friendly and practical AI travel agent.

Your only job in this step is to RESEARCH and produce a DRAFT itinerary.

DATE HANDLING — important:
- Accept any date format the user provides: a month and year (e.g. "September 2026"),
  a range (e.g. "June 15-30"), or approximate dates (e.g. "late October")
- Never ask the user to clarify or reformat their dates
- If only a month and year are given, assume travel spans the full duration they specified
  and use that month/year in the itinerary header

RESEARCH RULES — follow these before writing:
- Call search_web to find real restaurants, hotels, and attractions for the destination
- Search once per category (e.g. "budget hotels Lisbon", "best pastry shops Lisbon")
- Only include businesses that appeared in search results — do not invent names

DRAFT OUTPUT FORMAT:
- Trip overview: destination, dates, travelers, budget, style
- Day-by-day plan with morning / afternoon / evening slots
- Each entry: business name, why it's recommended, approx cost
- Estimated per-day budget

Write the full draft itinerary now. Label it clearly: DRAFT ITINERARY""")


CLOUD_CRITIC_PROMPT = SystemMessage(content="""You are a travel itinerary critic. You will receive a DRAFT ITINERARY.

Your job is to review it and provide a SHORT, SPECIFIC list of improvements.

Check for these issues only:
1. Pacing — are too many stops crammed into one day?
2. Realism — are travel times between locations accounted for?
3. Budget consistency — do the recommendations match the stated budget?
4. Completeness — are any obvious gaps missing (e.g. no dinner on day 2)?
5. Generic entries — flag any business names that seem invented or unverified

Output format — use this exact structure:
CRITIQUE:
- [issue 1]
- [issue 2]
- [issue 3]
(maximum 5 bullet points, be specific and actionable)""")


CLOUD_FINAL_PROMPT = """You are VoyageGT. You have a DRAFT ITINERARY and a CRITIQUE.

Your job: apply the critique's fixes and write the final, polished itinerary.

Rules:
- Fix every issue raised in the critique
- Do not change things the critique did not flag
- Keep all real business names from the draft
- Output only the final itinerary — no preamble, no critique summary

Write the FINAL ITINERARY now:"""


def cloud_agent_node(state: MessagesState):
    """
    Step 1 of 3 for cloud providers.
    Always prepends the system prompt regardless of conversation length,
    and includes the full message history so context is never lost.
    """
    messages = [CLOUD_AGENT_PROMPT] + state["messages"]
    response = llm_agent.invoke(messages)
    return {"messages": [response]}


def cloud_critic_node(state: MessagesState):
    """
    Step 2 of 3 for cloud providers.
    Extracts the draft by taking the last substantive AI text in state,
    regardless of whether it contains a 'DRAFT ITINERARY' label.
    """
    draft_content = ""

    for msg in reversed(state["messages"]):
        # Skip tool call messages and tool result messages
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            continue
        if getattr(msg, "type", "") == "tool":
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, str) and len(content) > 200:
            draft_content = content
            break

    if not draft_content:
        return {"messages": [AIMessage(content="CRITIQUE:\n- Could not locate a draft itinerary to critique.")]}

    critique_messages = [
        CLOUD_CRITIC_PROMPT,
        HumanMessage(content=f"Please critique this draft itinerary:\n\n{draft_content}"),
    ]
    critique = llm_critic.invoke(critique_messages)
    if not critique.content.strip().startswith("CRITIQUE"):
        critique.content = "CRITIQUE:\n" + critique.content
    return {"messages": [critique]}


def cloud_final_agent_node(state: MessagesState):
    """
    Step 3 of 3 for cloud providers.
    Extracts draft and critique by walking state in reverse:
    - critique is the most recent short-ish AI message starting with CRITIQUE
    - draft is the most recent long AI message that isn't the critique
    Falls back to positional extraction if labels are missing.
    """
    draft_content = ""
    critique_content = ""

    # Collect all substantive AI text messages, newest first
    ai_texts = []
    for msg in reversed(state["messages"]):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            continue
        if getattr(msg, "type", "") == "tool":
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, str) and len(content) > 50:
            ai_texts.append(content)

    # First pass: try label-based extraction
    for text in ai_texts:
        if text.strip().startswith("CRITIQUE") and not critique_content:
            critique_content = text
        elif len(text) > 200 and not draft_content and not text.strip().startswith("CRITIQUE"):
            draft_content = text
        if draft_content and critique_content:
            break

    # Second pass: fall back to positional — newest = critique, second newest = draft
    if not critique_content and len(ai_texts) >= 1:
        critique_content = ai_texts[0]
    if not draft_content and len(ai_texts) >= 2:
        draft_content = ai_texts[1]

    final_messages = [
        HumanMessage(content=(
            f"{CLOUD_FINAL_PROMPT}\n\n"
            f"--- DRAFT ITINERARY ---\n{draft_content}\n\n"
            f"--- CRITIQUE ---\n{critique_content}"
        ))
    ]
    response = llm_final.invoke(final_messages)
    return {"messages": [response]}


# ==============================================================
# GRAPH — two separate paths wired by MODEL_PROVIDER
# ==============================================================

graph_builder = StateGraph(MessagesState)

if MODEL_PROVIDER == "local":
    graph_builder.add_node("agent", local_agent_node)
    graph_builder.add_node("tools", ToolNode(tools))

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END}
    )
    graph_builder.add_edge("tools", "agent")

else:  # cloud
    graph_builder.add_node("agent", cloud_agent_node)
    graph_builder.add_node("critic", cloud_critic_node)
    graph_builder.add_node("final_agent", cloud_final_agent_node)
    graph_builder.add_node("tools", ToolNode(tools))

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: "critic"}
    )
    graph_builder.add_edge("tools", "agent")
    graph_builder.add_edge("critic", "final_agent")
    graph_builder.add_edge("final_agent", END)

voyage_agent = graph_builder.compile()