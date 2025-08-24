# app.py
# Streamlit + LangGraph RAG over a PDF with:
# 1) history-aware question condensation
# 2) secondary reranking
# 3) a hallucination guard that abstains when evidence is weak

import os
import numpy as np
import streamlit as st
from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass

# LangChain / LangGraph
try:
    from langchain_community.document_loaders import PyMuPDFLoader as PreferredLoader
    _LOADER_NAME = "PyMuPDF"
except Exception:
    from langchain_community.document_loaders import PyPDFLoader as PreferredLoader
    _LOADER_NAME = "PyPDF"
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL   = "gpt-4o"

# Retrieval knobs (top of app.py)
TOP_K_INITIAL = 40      # aggressive recall for specs/tables
TOP_K_FINAL   = 9       # keep more evidence
SIM_THRESHOLD = 0.35    # stricter evidence gate for codebook-style PDFs

st.set_page_config(page_title="LangGraph PDF Chabot", layout="wide")

# -----------------------------
# LangGraph State
# -----------------------------
class GraphState(TypedDict):
    query: str                         # raw user query
    condensed_query: str               # history-aware rewritten query
    history: List[Dict[str, str]]      # [{"role": "user/assistant", "content": "..."}]
    candidates: List[Document]         # initial retrieved docs (k=TOP_K_INITIAL)
    context: List[Document]            # reranked, top-k final docs
    scores: List[float]                # cosine scores for `context`
    guard_pass: bool                   # hallucination guard decision
    answer: str                        # final model output (or abstain message)

# -----------------------------
# Helpers: Build/Cache the Vector Store
# -----------------------------
@dataclass
class RAGIndex:
    vs: FAISS
    retriever: Any

def build_index_from_pdf(file_bytes: bytes, file_name: str, embeddings: OpenAIEmbeddings) -> RAGIndex:
    # write upload to disk
    with open(file_name, "wb") as f:
        f.write(file_bytes)

    # load pages (keep using your PreferredLoader)
    loader = PreferredLoader(file_name)
    pages = loader.load()  # one Document per physical page
    page_count = len(pages)

    # get page labels via PyMuPDF; fallback to 1-based numbers if unavailable
    labels = [str(i + 1) for i in range(page_count)]
    if fitz is not None:
        try:
            doc = fitz.open(file_name)
            labels = []
            for i in range(page_count):
                # returns logical label if present; else None
                lab = doc[i].get_label() if hasattr(doc[i], "get_label") else None
                labels.append(lab if lab else str(i + 1))
            doc.close()
        except Exception:
            pass  # keep numeric fallback

    # two-pass chunking tuned for specs/tables
    text_splitter  = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    table_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=40)

    def is_table_like(txt: str) -> bool:
        n = max(len(txt), 1)
        delims = txt.count("|") + txt.count("\t") + txt.count(",")
        digit_ratio = sum(ch.isdigit() for ch in txt) / n
        delim_ratio = delims / n
        return ("Table" in txt[:120]) or (digit_ratio > 0.18) or (delim_ratio > 0.02)

    chunks: List[Document] = []
    cid = 0
    for page_idx, d in enumerate(pages):
        page_label = labels[page_idx]
        splitter = table_splitter if is_table_like(d.page_content) else text_splitter

        # split while preserving metadata
        parts = splitter.split_documents([d])
        for p in parts:
            p.metadata["source"] = os.path.basename(file_name)
            p.metadata["page_index"] = page_idx            # 0-based physical index
            p.metadata["page_label"] = str(page_label)     # human-facing label (roman / arabic)
            p.metadata["chunk_id"] = cid
            cid += 1
            chunks.append(p)

    vs = FAISS.from_documents(chunks, embeddings)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K_INITIAL})
    return RAGIndex(vs=vs, retriever=retriever)



# -----------------------------
# Nodes
# -----------------------------
def condense_question_node(state: GraphState, llm: ChatOpenAI) -> GraphState:
    """Rewrite follow-ups into a self-contained query using chat history."""
    system = (
        "You rewrite the user's latest question into a single, self-contained query "
        "using the conversation history. Do not answer; only rewrite. Keep it concise."
    )
    messages = [{"role": "system", "content": system}]
    # feed a short window to keep prompts lean
    for m in state["history"][-8:]:
        messages.append(m)
    messages.append({"role": "user", "content": f"Rewrite this into a standalone query:\n\n{state['query']}"})
    resp = llm.invoke(messages)
    state["condensed_query"] = resp.content.strip() or state["query"]
    return state

def retrieve_node(state: GraphState, retriever) -> GraphState:
    q = state.get("condensed_query") or state["query"]
    docs = retriever.get_relevant_documents(q)
    state["candidates"] = docs
    return state

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def rerank_node(state: GraphState, embed: OpenAIEmbeddings) -> GraphState:
    """Secondary rerank with cosine similarity between the condensed query and each candidate."""
    q_text = state.get("condensed_query") or state["query"]
    if not state.get("candidates"):
        state["context"] = []
        state["scores"] = []
        return state

    # Embed query once
    q_vec = np.array(embed.embed_query(q_text), dtype=np.float32)

    # Embed each candidate chunk (use page_content only)
    cand_vecs: List[np.ndarray] = []
    for d in state["candidates"]:
        cand_vec = np.array(embed.embed_query(d.page_content), dtype=np.float32)  # using query endpoint is fine here
        cand_vecs.append(cand_vec)

    # Score and sort
    scored = []
    for d, v in zip(state["candidates"], cand_vecs):
        scored.append((d, _cosine(q_vec, v)))

    scored.sort(key=lambda t: t[1], reverse=True)
    top = scored[:TOP_K_FINAL]
    state["context"] = [d for d, s in top]
    state["scores"]  = [s for d, s in top]
    return state

def guard_node(state: GraphState) -> GraphState:
    """Abstain if evidence is weak (no doc passes SIM_THRESHOLD)."""
    best = max(state["scores"]) if state.get("scores") else 0.0
    state["guard_pass"] = bool(best >= SIM_THRESHOLD)
    if not state["guard_pass"]:
        state["answer"] = (
            "I don’t have sufficient evidence in the document to answer that precisely. "
            "Try rephrasing or pointing me to a section/page."
        )
    return state

def generate_node(state: GraphState, llm: ChatOpenAI) -> GraphState:
    system = (
        "You are a precise RAG assistant. Answer ONLY from the provided context. "
        "If the answer isn't in the context, say you don't know. "
        "Always give reference to section you found the answer from, and any other sections that are referred to in that section."
        "If the context of the text which contains the answer refers to an image, provide the image and page number of the image."
        "Include inline citations like [p. X], use the page numbers found in the document."
    )

    def doc_snippet(d: Document) -> str:
        label = d.metadata.get("page_label") or str((d.metadata.get("page_index") or 0) + 1)
        src   = d.metadata.get("source", "pdf")
        return f"(source: {src}, p. {label})\n{d.page_content}"

    context_block = "\n\n---\n".join(doc_snippet(d) for d in state["context"])
    messages = [
        {"role": "system", "content": system},
        *state["history"][-8:],  # short history window for style/continuity; retrieval used condensed_query already
        {"role": "user", "content": f"Question: {state.get('condensed_query') or state['query']}\n\nContext:\n{context_block}"}
    ]

    resp = llm.invoke(messages)
    state["answer"] = resp.content
    return state

# -----------------------------
# Build the graph (wired once)
# -----------------------------
def build_graph(retriever, llm, emb):
    g = StateGraph(GraphState)

    g.add_node("condense", lambda s: condense_question_node(s, llm))
    g.add_node("retrieve", lambda s: retrieve_node(s, retriever))
    g.add_node("rerank",   lambda s: rerank_node(s, emb))
    g.add_node("guard",    guard_node)
    g.add_node("generate", lambda s: generate_node(s, llm))

    g.set_entry_point("condense")
    g.add_edge("condense", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "guard")

    # conditional branch based on hallucination guard
    def guard_route(state: GraphState):
        return "generate" if state.get("guard_pass") else END

    g.add_conditional_edges("guard", guard_route, {"generate": "generate"})
    g.add_edge("generate", END)

    return g.compile(checkpointer=MemorySaver())

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("LangGraph PDF Chatbot (history-aware + rerank mechanism + guardrails)")

with st.sidebar:
    st.subheader("Setup")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    build_btn = st.button("Build/Reset Index")

    st.markdown("---")
    st.caption("This app rewrites follow-ups, reranks results, and abstains if evidence is weak.")

if "index" not in st.session_state:
    st.session_state.index = None
if "graph" not in st.session_state:
    st.session_state.graph = None
if "history" not in st.session_state:
    st.session_state.history = []

if not api_key:
    st.info("Enter your OpenAI API key in the sidebar to begin.")
    st.stop()

emb = OpenAIEmbeddings(api_key=api_key, model=EMBED_MODEL)
llm = ChatOpenAI(api_key=api_key, model=GPT_MODEL, temperature=0)

if build_btn:
    st.session_state.index = None
    st.session_state.graph = None
    st.session_state.history = []

if uploaded and (st.session_state.index is None):
    with st.spinner("Building vector store from PDF…"):
        st.session_state.index = build_index_from_pdf(uploaded.read(), uploaded.name, emb)
        st.success("Index built.")

if st.session_state.index and st.session_state.graph is None:
    st.session_state.graph = build_graph(st.session_state.index.retriever, llm, emb)

if st.session_state.index is None or st.session_state.graph is None:
    st.warning("Upload a PDF and build the index to start chatting.")
    st.stop()

# Render history
for m in st.session_state.history:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.write(m["content"])

user_msg = st.chat_input("Ask something grounded in the PDF…")
if user_msg:
    st.session_state.history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            state: GraphState = {
                "query": user_msg,
                "condensed_query": "",
                "history": st.session_state.history[-10:],
                "candidates": [],
                "context": [],
                "scores": [],
                "guard_pass": False,
                "answer": ""
            }
            result = st.session_state.graph.invoke(state, config={"configurable": {"thread_id": "session"}})

            st.write(result["answer"])

            # Sources + scores
            if result.get("context"):
                with st.expander("Sources"):
                    for d, s in zip(result["context"], result.get("scores", [])):
                        label = d.metadata.get("page_label") or str((d.metadata.get("page_index") or 0) + 1)
                        st.markdown(
                            f"- **{d.metadata.get('source','pdf')}**, page **{label}**, "
                            f"chunk `{d.metadata.get('chunk_id','?')}`, similarity `{s:.3f}`"
                        )


    st.session_state.history.append({"role": "assistant", "content": result["answer"]})
