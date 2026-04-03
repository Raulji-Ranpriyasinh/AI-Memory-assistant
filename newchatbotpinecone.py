import os
import uuid
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore

# Pinecone + Embeddings
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────────
# Database & Pinecone Configuration
# ─────────────────────────────────────────────
DB_URI = os.getenv("DB_URI", "postgresql://postgres:postgres@localhost:5442/postgres")

PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX", "ltm-memories")
PINECONE_CLOUD     = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION    = os.getenv("PINECONE_REGION", "us-east-1")

# Embedding model — all-MiniLM-L6-v2 outputs 384-dim vectors
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM        = 384

# ─────────────────────────────────────────────
# Memory-layer Constants
# ─────────────────────────────────────────────
STM_A_WINDOW_SIZE        = 10    # Recent raw turns kept in state
SUMMARY_LAYER_THRESHOLDS = [20, 50, 100]
LTM_SALIENCE_THRESHOLD   = 0.7   # Min score before writing to LTM
LTM_TOP_K                = 20    # How many memories to retrieve per query


# ─────────────────────────────────────────────
# System Prompts
# ─────────────────────────────────────────────
CHAT_SYSTEM_PROMPT = """You are a helpful assistant with advanced memory capabilities.

You have access to:
1. Recent conversation context (last few exchanges)
2. Summarized conversation history (if available)
3. Long-term user memories (persistent facts and preferences)

Your goal is to provide relevant, friendly, and tailored assistance.

PERSONALIZATION GUIDELINES:
- If the user's name is known, address them by name
- Reference known projects, tools, or preferences
- Adjust tone to feel friendly and natural
- Only personalize based on known details, never assume

CURRENT USER CONTEXT:
{user_context}

CONVERSATION SUMMARIES:
{conversation_summaries}

LONG-TERM MEMORIES (retrieved via semantic search):
{ltm_content}
"""

CANDIDATE_EXTRACTION_PROMPT = """Extract memory-worthy information from the recent conversation.

RECENT CONVERSATION:
{recent_messages}

TASK:
Identify facts worth storing long-term. Categorize each into:
- identity: Name, location, profession, personal identifiers
- preferences: Likes, dislikes, habits, communication style
- projects: Current work, goals, ongoing activities
- facts: Other stable factual information

For each item:
1. Write as a concise atomic sentence
2. Only extract explicit information (no speculation)
3. Include source context if relevant

Return ONLY facts that are:
- Stable over time (not ephemeral)
- User-specific (not general knowledge)
- Actionable for personalization
"""

SCORING_PROMPT = """Score the salience (long-term importance) of each memory candidate.

CANDIDATES:
{candidates}

EXISTING MEMORIES:
{existing_memories}

TASK:
For each candidate, assign:
- salience_score (0.0-1.0): How important for long-term storage
- is_duplicate (bool): Whether substantially covered by existing memories
- reasoning (str): Brief explanation

High salience (0.8-1.0): Core identity, strong preferences, major projects
Medium salience (0.5-0.7): Useful context, minor preferences
Low salience (0.0-0.4): Ephemeral, already covered, or not actionable
"""

SUMMARY_GENERATION_PROMPT = """Generate a concise summary of the conversation segment.

CONVERSATION SEGMENT:
{messages}

PREVIOUS SUMMARY (if any):
{previous_summary}

TASK:
Create a structured JSON summary with:
- key_topics: Main discussion topics
- decisions_made: Any decisions or conclusions
- action_items: Tasks or follow-ups mentioned
- important_context: Critical context for future reference

Keep it concise but preserve essential information.
"""

SUMMARY_MERGE_PROMPT = """Merge multiple conversation summaries into a coherent overview.

SUMMARIES TO MERGE:
{summaries}

TASK:
Create a unified summary that:
- Preserves chronological flow
- Highlights key themes
- Maintains important details
- Removes redundancy

Return structured JSON with the same format as individual summaries.
"""


# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────
class MemoryCandidate(BaseModel):
    text: str = Field(description="The memory content")
    category: str = Field(description="identity, preferences, projects, or facts")
    context: Optional[str] = Field(default=None, description="Additional context")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ScoredCandidate(BaseModel):
    candidate: MemoryCandidate
    salience_score: float = Field(ge=0.0, le=1.0)
    is_duplicate: bool
    reasoning: str


class CandidateExtractionResult(BaseModel):
    candidates: List[MemoryCandidate] = Field(default_factory=list)


class ScoringResult(BaseModel):
    scored_candidates: List[ScoredCandidate] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    key_topics: List[str] = Field(default_factory=list)
    decisions_made: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    important_context: str = ""
    turn_range: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class MergedSummary(BaseModel):
    key_topics: List[str] = Field(default_factory=list)
    decisions_made: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    important_context: str = ""
    chronological_flow: str = ""


# ─────────────────────────────────────────────
# Pinecone LTM Manager
# ─────────────────────────────────────────────
class PineconeLTMManager:
    """
    Handles all Long-Term Memory operations via Pinecone.

    Embeddings are generated with the all-MiniLM-L6-v2 model
    (384-dimensional dense vectors, fast CPU inference).

    Index layout
    ─────────────
    Each vector ID  : "<user_id>__<uuid4>"
    Metadata stored : text, category, context, salience_score, timestamp, user_id

    Namespaces are NOT used so that a single serverless index can serve
    all users; filtering is done via metadata field 'user_id'.
    """

    def __init__(self):
        if not PINECONE_API_KEY:
            raise ValueError(
                "PINECONE_API_KEY not found. "
                "Add it to your .env file or environment."
            )

        # Embedding model (downloads ~90 MB on first run, then cached)
        print("🔧 Loading all-MiniLM-L6-v2 embedding model …")
        self._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("✅ Embedding model loaded.")

        # Pinecone client & index
        self._pc = Pinecone(api_key=PINECONE_API_KEY)
        self._index = self._get_or_create_index()

    # ── Index management ──────────────────────────────────────────────────

    def _get_or_create_index(self):
        """Return the Pinecone index, creating it if necessary."""
        existing_names = [idx.name for idx in self._pc.list_indexes()]

        if PINECONE_INDEX not in existing_names:
            print(f"📦 Creating Pinecone index '{PINECONE_INDEX}' …")
            self._pc.create_index(
                name=PINECONE_INDEX,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION,
                ),
            )
            print(f"✅ Index '{PINECONE_INDEX}' created.")
        else:
            print(f"✅ Connected to existing Pinecone index '{PINECONE_INDEX}'.")

        return self._pc.Index(PINECONE_INDEX)

    # ── Embedding helpers ─────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        """Return a 384-dim embedding for a single text string."""
        return self._embedder.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a list of texts."""
        return self._embedder.encode(texts, convert_to_numpy=True).tolist()

    # ── Write ─────────────────────────────────────────────────────────────

    def write_memory(
        self,
        user_id: str,
        text: str,
        category: str,
        salience_score: float,
        context: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> str:
        """
        Embed *text* and upsert it into Pinecone.
        Returns the generated vector ID.
        """
        memory_id  = f"{user_id}__{uuid.uuid4()}"
        vector     = self.embed(text)
        ts         = timestamp or datetime.now().isoformat()

        metadata = {
            "user_id":        user_id,
            "text":           text,
            "category":       category,
            "salience_score": salience_score,
            "context":        context or "",
            "timestamp":      ts,
        }

        self._index.upsert(vectors=[(memory_id, vector, metadata)])
        return memory_id

    def write_memories_batch(
        self,
        user_id: str,
        scored_candidates: List[ScoredCandidate],
    ) -> List[str]:
        """
        Batch-upsert all high-salience, non-duplicate candidates.
        Returns the list of written IDs.
        """
        to_write = [
            sc for sc in scored_candidates
            if sc.salience_score >= LTM_SALIENCE_THRESHOLD and not sc.is_duplicate
        ]
        if not to_write:
            return []

        texts   = [sc.candidate.text for sc in to_write]
        vectors = self.embed_batch(texts)

        upsert_payload: List[tuple] = []
        written_ids: List[str] = []

        for sc, vec in zip(to_write, vectors):
            memory_id = f"{user_id}__{uuid.uuid4()}"
            written_ids.append(memory_id)
            metadata = {
                "user_id":        user_id,
                "text":           sc.candidate.text,
                "category":       sc.candidate.category,
                "salience_score": sc.salience_score,
                "context":        sc.candidate.context or "",
                "timestamp":      sc.candidate.timestamp,
            }
            upsert_payload.append((memory_id, vec, metadata))

        # Pinecone recommends batches ≤ 100 vectors
        for i in range(0, len(upsert_payload), 100):
            self._index.upsert(vectors=upsert_payload[i : i + 100])

        return written_ids

    # ── Retrieval ─────────────────────────────────────────────────────────

    def search_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = LTM_TOP_K,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search: embed *query* and return the closest LTM memories
        for *user_id*, optionally filtered by category.
        Results are sorted by (salience_score DESC, relevance_score DESC).
        """
        query_vec = self.embed(query)

        pinecone_filter: Dict[str, Any] = {"user_id": {"$eq": user_id}}
        if category_filter:
            pinecone_filter["category"] = {"$eq": category_filter}

        response = self._index.query(
            vector=query_vec,
            top_k=top_k,
            filter=pinecone_filter,
            include_metadata=True,
        )

        memories: List[Dict[str, Any]] = []
        for match in response.matches:
            meta = match.metadata or {}
            memories.append(
                {
                    "id":             match.id,
                    "text":           meta.get("text", ""),
                    "category":       meta.get("category", "fact"),
                    "salience_score": meta.get("salience_score", 0.0),
                    "context":        meta.get("context", ""),
                    "timestamp":      meta.get("timestamp", ""),
                    "relevance_score": match.score,  # cosine similarity
                }
            )

        # Primary sort: salience; secondary: semantic relevance
        memories.sort(
            key=lambda m: (m["salience_score"], m["relevance_score"]),
            reverse=True,
        )
        return memories

    def list_all_memories(
        self, user_id: str, top_k: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Retrieve ALL memories for a user (no query text).
        Uses a zero-vector query — works well for small-to-medium indexes.
        """
        zero_vec = [0.0] * EMBEDDING_DIM
        response = self._index.query(
            vector=zero_vec,
            top_k=top_k,
            filter={"user_id": {"$eq": user_id}},
            include_metadata=True,
        )
        memories = []
        for match in response.matches:
            meta = match.metadata or {}
            memories.append(
                {
                    "id":             match.id,
                    "text":           meta.get("text", ""),
                    "category":       meta.get("category", "fact"),
                    "salience_score": meta.get("salience_score", 0.0),
                    "context":        meta.get("context", ""),
                    "timestamp":      meta.get("timestamp", ""),
                }
            )
        memories.sort(key=lambda m: m["salience_score"], reverse=True)
        return memories

    def delete_memory(self, memory_id: str):
        """Delete a specific memory by its vector ID."""
        self._index.delete(ids=[memory_id])

    def delete_user_memories(self, user_id: str):
        """Delete ALL memories for a user (uses metadata filter delete)."""
        self._index.delete(filter={"user_id": {"$eq": user_id}})


# ─────────────────────────────────────────────
# Memory Controller
# ─────────────────────────────────────────────
class MemoryController:
    """Orchestrates all memory operations (STM-A/B/C + Pinecone LTM)."""

    def __init__(self, db_uri: str):
        self.db_uri = db_uri

        # LLMs
        self.chat_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=0.7
        )
        self.memory_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=0
        )

        # Structured extractors
        self.candidate_extractor = self.memory_llm.with_structured_output(
            CandidateExtractionResult
        )
        self.scorer    = self.memory_llm.with_structured_output(ScoringResult)
        self.summarizer = self.memory_llm.with_structured_output(ConversationSummary)
        self.merger    = self.memory_llm.with_structured_output(MergedSummary)

        # Pinecone LTM
        self.ltm = PineconeLTMManager()

    # ── STM-A ─────────────────────────────────────────────────────────────

    def get_stm_a_raw(
        self, all_messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        """Return the raw recent-window slice (last N messages)."""
        return (
            all_messages[-STM_A_WINDOW_SIZE:]
            if len(all_messages) > STM_A_WINDOW_SIZE
            else all_messages
        )

    # ── STM-B ─────────────────────────────────────────────────────────────

    def get_stm_b_summaries(
        self, store: BaseStore, user_id: str, thread_id: str
    ) -> List[ConversationSummary]:
        ns    = ("summaries", user_id, thread_id)
        items = store.search(ns)
        summaries = []
        for item in items:
            try:
                summaries.append(ConversationSummary(**item.value.get("data", {})))
            except Exception:
                continue
        return sorted(summaries, key=lambda s: s.timestamp)

    # ── STM-C ─────────────────────────────────────────────────────────────

    def extract_candidates_stm_c(
        self, messages: List[BaseMessage]
    ) -> List[MemoryCandidate]:
        if len(messages) < 2:
            return []

        recent_text = "\n".join(
            [
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in messages[-6:]
            ]
        )
        try:
            result: CandidateExtractionResult = self.candidate_extractor.invoke(
                [
                    SystemMessage(
                        content=CANDIDATE_EXTRACTION_PROMPT.format(
                            recent_messages=recent_text
                        )
                    ),
                    HumanMessage(content="Extract memory candidates."),
                ]
            )
            return result.candidates
        except Exception as e:
            print(f"Warning: Candidate extraction failed: {e}")
            return []

    # ── Scoring ───────────────────────────────────────────────────────────

    def score_candidates(
        self,
        candidates: List[MemoryCandidate],
        existing_memories: List[str],
    ) -> List[ScoredCandidate]:
        if not candidates:
            return []

        candidates_text = "\n".join(
            [f"{i+1}. [{c.category}] {c.text}" for i, c in enumerate(candidates)]
        )
        existing_text = "\n".join(existing_memories) if existing_memories else "(none)"

        try:
            result: ScoringResult = self.scorer.invoke(
                [
                    SystemMessage(
                        content=SCORING_PROMPT.format(
                            candidates=candidates_text,
                            existing_memories=existing_text,
                        )
                    ),
                    HumanMessage(content="Score these candidates."),
                ]
            )
            return result.scored_candidates
        except Exception as e:
            print(f"Warning: Candidate scoring failed: {e}")
            return []

    # ── LTM helpers (delegating to Pinecone) ──────────────────────────────

    def write_to_ltm(
        self,
        user_id: str,
        scored_candidates: List[ScoredCandidate],
    ):
        """Write high-salience memories to Pinecone in batch."""
        written = self.ltm.write_memories_batch(user_id, scored_candidates)
        if written:
            print(f"  💾 Wrote {len(written)} memories to Pinecone LTM.")

    def get_ltm_memories_semantic(
        self,
        user_id: str,
        query: str,
        top_k: int = LTM_TOP_K,
    ) -> List[Dict[str, Any]]:
        """Semantic retrieval — most relevant memories for the current query."""
        return self.ltm.search_memories(user_id, query, top_k=top_k)

    def get_all_ltm_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Full memory dump for '/memories' command."""
        return self.ltm.list_all_memories(user_id)

    # ── Summarization ─────────────────────────────────────────────────────

    def generate_summary(
        self,
        messages: List[BaseMessage],
        turn_range: str,
        previous_summary: Optional[ConversationSummary] = None,
    ) -> ConversationSummary:
        messages_text = "\n".join(
            [
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in messages
            ]
        )
        prev_text = (
            json.dumps(previous_summary.model_dump(), indent=2)
            if previous_summary
            else ""
        )
        try:
            summary: ConversationSummary = self.summarizer.invoke(
                [
                    SystemMessage(
                        content=SUMMARY_GENERATION_PROMPT.format(
                            messages=messages_text,
                            previous_summary=prev_text,
                        )
                    ),
                    HumanMessage(content="Generate summary."),
                ]
            )
            summary.turn_range = turn_range
            return summary
        except Exception as e:
            print(f"Warning: Summary generation failed: {e}")
            return ConversationSummary(turn_range=turn_range)

    def merge_summaries(
        self, summaries: List[ConversationSummary]
    ) -> MergedSummary:
        if not summaries:
            return MergedSummary()

        summaries_text = "\n\n".join(
            [
                f"Summary ({s.turn_range}):\n{json.dumps(s.model_dump(), indent=2)}"
                for s in summaries
            ]
        )
        try:
            merged: MergedSummary = self.merger.invoke(
                [
                    SystemMessage(
                        content=SUMMARY_MERGE_PROMPT.format(summaries=summaries_text)
                    ),
                    HumanMessage(content="Merge summaries."),
                ]
            )
            return merged
        except Exception as e:
            print(f"Warning: Summary merge failed: {e}")
            return MergedSummary()

    def update_summaries(
        self,
        store: BaseStore,
        user_id: str,
        thread_id: str,
        all_messages: List[BaseMessage],
    ):
        total_turns = len(all_messages) // 2
        for threshold in SUMMARY_LAYER_THRESHOLDS:
            if total_turns >= threshold:
                ns       = ("summaries", user_id, thread_id)
                existing = store.search(ns)
                has_summary = any(
                    str(threshold) in item.value.get("data", {}).get("turn_range", "")
                    for item in existing
                )
                if not has_summary:
                    start_idx = max(0, threshold - 20) * 2
                    end_idx   = threshold * 2
                    segment   = all_messages[start_idx:end_idx]
                    if segment:
                        summary = self.generate_summary(
                            segment, f"turns {threshold-20}-{threshold}"
                        )
                        store.put(ns, f"summary_{threshold}", {"data": summary.model_dump()})


# ─────────────────────────────────────────────
# Multi-Layer Chatbot
# ─────────────────────────────────────────────
class MultiLayerChatbot:
    """Main chatbot with multi-layer memory (STM-A/B/C + Pinecone LTM)."""

    def __init__(self, user_id: str = "default_user"):
        self.user_id    = user_id
        self.db_uri     = DB_URI
        self.controller = MemoryController(DB_URI)
        self.graph      = self._build_graph()

    # ── Graph construction ────────────────────────────────────────────────

    def _build_graph(self):
        builder = StateGraph(MessagesState)

        builder.add_node("stm_a_update",  self._stm_a_node)
        builder.add_node("stm_b_update",  self._stm_b_node)
        builder.add_node("stm_c_extract", self._stm_c_node)
        builder.add_node("ltm_gate",      self._ltm_gate_node)
        builder.add_node("chat",          self._chat_node)

        builder.add_edge(START,          "stm_a_update")
        builder.add_edge("stm_a_update", "stm_b_update")
        builder.add_edge("stm_b_update", "stm_c_extract")
        builder.add_edge("stm_c_extract","ltm_gate")
        builder.add_edge("ltm_gate",     "chat")
        builder.add_edge("chat",          END)

        # One-time DB setup
        with PostgresSaver.from_conn_string(self.db_uri) as cp:
            cp.setup()
        with PostgresStore.from_conn_string(self.db_uri) as st:
            st.setup()

        return builder

    # ── Graph nodes ───────────────────────────────────────────────────────

    def _stm_a_node(
        self, state: MessagesState, config: RunnableConfig, *, store: BaseStore
    ):
        """STM-A: pass-through (window extracted on demand)."""
        return {}

    def _stm_b_node(
        self, state: MessagesState, config: RunnableConfig, *, store: BaseStore
    ):
        """STM-B: update multi-layer summaries in Postgres."""
        user_id   = config["configurable"]["user_id"]
        thread_id = config["configurable"]["thread_id"]
        self.controller.update_summaries(
            store, user_id, thread_id, state["messages"]
        )
        return {}

    def _stm_c_node(
        self, state: MessagesState, config: RunnableConfig, *, store: BaseStore
    ):
        """STM-C: extract typed candidates and buffer them in Postgres."""
        candidates = self.controller.extract_candidates_stm_c(state["messages"])
        if candidates:
            user_id   = config["configurable"]["user_id"]
            thread_id = config["configurable"]["thread_id"]
            ns = ("candidates", user_id, thread_id)
            for c in candidates:
                store.put(ns, str(uuid.uuid4()), {"data": c.model_dump()})
        return {}

    def _ltm_gate_node(
        self, state: MessagesState, config: RunnableConfig, *, store: BaseStore
    ):
        """
        LTM Gate: score buffered candidates, write high-salience ones
        to Pinecone, then clear the Postgres candidate buffer.
        """
        user_id   = config["configurable"]["user_id"]
        thread_id = config["configurable"]["thread_id"]

        # Load candidates from buffer
        ns             = ("candidates", user_id, thread_id)
        candidate_items = store.search(ns)

        candidates: List[MemoryCandidate] = []
        for item in candidate_items:
            try:
                candidates.append(MemoryCandidate(**item.value.get("data", {})))
            except Exception:
                continue

        if not candidates:
            return {}

        # Fetch existing LTM texts for dedup check (semantic search on the
        # first candidate's text as a representative query)
        existing = self.controller.get_ltm_memories_semantic(
            user_id, candidates[0].text, top_k=30
        )
        existing_texts = [m.get("text", "") for m in existing]

        # Score & write to Pinecone
        scored = self.controller.score_candidates(candidates, existing_texts)
        self.controller.write_to_ltm(user_id, scored)

        # Clear Postgres buffer
        for item in candidate_items:
            store.delete(ns, item.key)

        return {}

    def _chat_node(
        self, state: MessagesState, config: RunnableConfig, *, store: BaseStore
    ):
        """
        Chat: assemble context from all memory layers and generate response.

        LTM is retrieved semantically using the user's latest message
        so the most relevant memories surface for each turn.
        """
        user_id   = config["configurable"]["user_id"]
        thread_id = config["configurable"]["thread_id"]

        # ── STM-A: raw recent window ──────────────────────────────────────
        stm_a = self.controller.get_stm_a_raw(state["messages"])

        # ── STM-B: summaries ─────────────────────────────────────────────
        stm_b          = self.controller.get_stm_b_summaries(store, user_id, thread_id)
        merged_summary = self.controller.merge_summaries(stm_b)
        summary_text   = (
            json.dumps(merged_summary.model_dump(), indent=2) if stm_b else "(none)"
        )

        # ── LTM: semantic retrieval using latest user message ────────────
        latest_user_msg = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        ltm_memories = self.controller.get_ltm_memories_semantic(
            user_id, latest_user_msg or "general context"
        )
        ltm_text = (
            "\n".join(
                [
                    f"[{m.get('category','fact')}] {m.get('text','')} "
                    f"(salience={m.get('salience_score',0):.2f}, "
                    f"relevance={m.get('relevance_score',0):.2f})"
                    for m in ltm_memories[:LTM_TOP_K]
                ]
            )
            if ltm_memories
            else "(none)"
        )

        # ── Build system prompt & respond ─────────────────────────────────
        system_msg = SystemMessage(
            content=CHAT_SYSTEM_PROMPT.format(
                user_context=f"User ID: {user_id}",
                conversation_summaries=summary_text,
                ltm_content=ltm_text,
            )
        )
        response = self.controller.chat_llm.invoke([system_msg] + stm_a)
        return {"messages": [response]}

    # ── Public API ────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """Send a message and return the assistant's reply."""
        config = {
            "configurable": {
                "user_id":   self.user_id,
                "thread_id": f"thread-{self.user_id}",
            }
        }
        with PostgresSaver.from_conn_string(self.db_uri) as checkpointer:
            with PostgresStore.from_conn_string(self.db_uri) as store:
                graph  = self.graph.compile(checkpointer=checkpointer, store=store)
                result = graph.invoke(
                    {"messages": [HumanMessage(content=user_message)]}, config
                )
                return result["messages"][-1].content

    def get_memories(self) -> Dict[str, Any]:
        """Return all LTM memories + conversation summaries."""
        ltm = self.controller.get_all_ltm_memories(self.user_id)

        thread_id = f"thread-{self.user_id}"
        with PostgresStore.from_conn_string(self.db_uri) as store:
            summaries = self.controller.get_stm_b_summaries(
                store, self.user_id, thread_id
            )

        return {
            "ltm":       ltm,
            "summaries": [s.model_dump() for s in summaries],
        }

    def search_memories(
        self,
        query: str,
        top_k: int = 10,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over LTM memories."""
        return self.controller.ltm.search_memories(
            self.user_id, query, top_k=top_k, category_filter=category
        )

    def get_conversation_history(self) -> List[dict]:
        """Retrieve recent conversation history (STM-A window)."""
        config = {
            "configurable": {"thread_id": f"thread-{self.user_id}"}
        }
        with PostgresSaver.from_conn_string(self.db_uri) as checkpointer:
            with PostgresStore.from_conn_string(self.db_uri) as store:
                graph = self.graph.compile(checkpointer=checkpointer, store=store)
                try:
                    state    = graph.get_state(config)
                    messages = state.values.get("messages", [])
                    stm_a    = self.controller.get_stm_a_raw(messages)
                    history  = []
                    for msg in stm_a:
                        if isinstance(msg, HumanMessage):
                            history.append({"role": "user", "content": msg.content})
                        elif isinstance(msg, AIMessage):
                            history.append({"role": "assistant", "content": msg.content})
                    return history
                except Exception:
                    return []


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("🤖 MULTI-LAYER MEMORY CHATBOT  (Pinecone LTM Edition)".center(70))
    print("=" * 70)
    print("\nMemory Architecture:")
    print("  📝 STM-A : Raw recent conversation window (last 10 turns)")
    print("  📊 STM-B : Multi-layer summaries in PostgreSQL (progressive)")
    print("  🎯 STM-C : Typed candidate buffer with salience scoring")
    print("  🧠 LTM   : Pinecone vector store  ← all-MiniLM-L6-v2 embeddings")
    print("  🚪 Gate  : Salience threshold + duplicate filter before LTM write")
    print("\n💎 Chat LLM  : Gemini 2.5 Flash")
    print("🔍 Embeddings: all-MiniLM-L6-v2 (384 dims, local inference)")
    print("📦 Vector DB : Pinecone Serverless")
    print("\nCommands:")
    print("  'bye' / 'exit'   - Exit the chatbot")
    print("  /memories        - View all LTM memories + summaries")
    print("  /history         - View recent conversation (STM-A)")
    print("  /summaries       - View detailed conversation summaries (STM-B)")
    print("  /search <query>  - Semantic search your LTM memories")
    print("=" * 70)

    user_id = input("\n👤 Enter your user ID (or press Enter for 'default_user'): ").strip()
    if not user_id:
        user_id = "default_user"

    print(f"\n✅ Logged in as: {user_id}")
    print("💬 Start chatting! (Type 'bye' to exit)\n")

    try:
        chatbot = MultiLayerChatbot(user_id=user_id)
    except Exception as e:
        print(f"❌ Error initialising chatbot: {e}")
        print("Check that PostgreSQL is running and PINECONE_API_KEY is set.")
        return

    while True:
        try:
            user_input = input(f"\n{user_id}> ").strip()
            if not user_input:
                continue

            # ── Exit ──────────────────────────────────────────────────────
            if user_input.lower() in ("bye", "exit", "quit"):
                print("\n👋 Goodbye! Your memories have been saved across all layers.")
                break

            # ── /memories ─────────────────────────────────────────────────
            if user_input == "/memories":
                memories = chatbot.get_memories()
                print("\n🧠 LONG-TERM MEMORIES (Pinecone):")
                ltm = memories.get("ltm", [])
                if ltm:
                    for i, mem in enumerate(ltm, 1):
                        print(
                            f"  {i}. [{mem.get('category','?')}] "
                            f"{mem.get('text','')} "
                            f"(salience={mem.get('salience_score',0):.2f})"
                        )
                else:
                    print("  (No LTM memories yet)")

                print("\n📊 CONVERSATION SUMMARIES (STM-B):")
                for i, s in enumerate(memories.get("summaries", []), 1):
                    topics = ", ".join(s.get("key_topics", []))
                    print(f"  {i}. {s.get('turn_range','?')}: {topics}")
                if not memories.get("summaries"):
                    print("  (No summaries yet)")
                continue

            # ── /history ──────────────────────────────────────────────────
            if user_input == "/history":
                history = chatbot.get_conversation_history()
                print("\n📝 RECENT CONVERSATION (STM-A):")
                if history:
                    for msg in history:
                        snippet = (
                            msg["content"][:100] + "…"
                            if len(msg["content"]) > 100
                            else msg["content"]
                        )
                        print(f"  [{msg['role'].upper()}] {snippet}")
                else:
                    print("  (No history yet)")
                continue

            # ── /summaries ────────────────────────────────────────────────
            if user_input == "/summaries":
                summaries = chatbot.get_memories().get("summaries", [])
                print("\n📊 DETAILED CONVERSATION SUMMARIES:")
                if summaries:
                    for i, s in enumerate(summaries, 1):
                        print(f"\n  Summary {i} ({s.get('turn_range','?')}):")
                        print(f"    Topics     : {', '.join(s.get('key_topics',[]))}")
                        print(f"    Decisions  : {', '.join(s.get('decisions_made',[])) or 'None'}")
                        print(f"    Action items: {', '.join(s.get('action_items',[])) or 'None'}")
                        if s.get("important_context"):
                            print(f"    Context    : {s['important_context']}")
                else:
                    print("  (No summaries yet)")
                continue

            # ── /search <query> ───────────────────────────────────────────
            if user_input.startswith("/search "):
                query   = user_input[len("/search "):].strip()
                results = chatbot.search_memories(query, top_k=10)
                print(f"\n🔍 Semantic search results for: '{query}'")
                if results:
                    for i, r in enumerate(results, 1):
                        print(
                            f"  {i}. [{r.get('category','?')}] {r.get('text','')} "
                            f"(salience={r.get('salience_score',0):.2f}, "
                            f"cosine={r.get('relevance_score',0):.3f})"
                        )
                else:
                    print("  (No results found)")
                continue

            # ── Normal chat ───────────────────────────────────────────────
            print("\n🤖 Assistant: ", end="", flush=True)
            response = chatbot.chat(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print("Please try again or type 'bye' to exit.")


if __name__ == "__main__":
    main()