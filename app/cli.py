"""
CLI entry point — all user-facing I/O lives here.
"""

from __future__ import annotations

import json

from app.chatbot import MultiLayerChatbot


def _banner() -> None:
    print("=" * 70)
    print("🤖 MULTI-LAYER MEMORY CHATBOT  (Production Edition)".center(70))
    print("=" * 70)
    print("\nMemory Architecture:")
    print("  📝 STM-A : Raw recent window (last 10 turns)")
    print("  📊 STM-B : Multi-layer summaries in PostgreSQL")
    print("  🎯 STM-C : Typed candidate buffer (extracted every 3 turns)")
    print("  🧠 LTM   : Pinecone — all-MiniLM-L6-v2 (384 dims)")
    print("  🚪 Gate  : Cosine pre-filter → LLM scorer → conflict resolver")
    print("\n💎 Chat model   : gemini-2.5-flash")
    print("⚡ Memory model  : gemini-2.0-flash-lite  (cheaper high-volume ops)")
    print("🔍 Embeddings   : all-MiniLM-L6-v2 (local CPU inference)")
    print("📦 Vector DB    : Pinecone Serverless")
    print("📊 Observability: structured JSONL metrics log")
    print("\nCommands:")
    print("  bye / exit       – Exit")
    print("  /memories        – View all LTM memories + summaries")
    print("  /history         – View recent conversation (STM-A)")
    print("  /summaries       – View detailed summaries (STM-B)")
    print("  /search <query>  – Semantic search LTM")
    print("  /metrics         – View observability metrics")
    print("  /prune           – Run memory decay & pruning")
    print("=" * 70)


def main() -> None:
    _banner()

    user_id = input("\n👤 Enter your user ID (Enter = 'default_user'): ").strip()
    if not user_id:
        user_id = "default_user"

    print(f"\n✅ Logged in as: {user_id}")
    print("💬 Start chatting!\n")

    try:
        chatbot = MultiLayerChatbot(user_id=user_id)
    except Exception as e:
        print(f"❌ Error initialising chatbot: {e}")
        print("Check PostgreSQL is running and PINECONE_API_KEY is set.")
        return

    while True:
        try:
            user_input = input(f"\n{user_id}> ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("bye", "exit", "quit"):
                print("\n👋 Goodbye! Memories saved across all layers.")
                break

            if user_input == "/memories":
                mems = chatbot.get_memories()
                print("\n🧠 LONG-TERM MEMORIES (Pinecone):")
                for i, m in enumerate(mems.get("ltm", []), 1):
                    print(
                        f"  {i}. [{m.get('category','?')}] {m.get('text','')} "
                        f"(salience={m.get('salience_score',0):.2f})"
                    )
                if not mems.get("ltm"):
                    print("  (none yet)")
                print("\n📊 SUMMARIES:")
                for i, s in enumerate(mems.get("summaries", []), 1):
                    print(f"  {i}. {s.get('turn_range','?')}: {', '.join(s.get('key_topics',[]))}")
                if not mems.get("summaries"):
                    print("  (none yet)")
                continue

            if user_input == "/history":
                history = chatbot.get_conversation_history()
                print("\n📝 RECENT CONVERSATION (STM-A):")
                for msg in history:
                    snippet = msg["content"][:100] + ("…" if len(msg["content"]) > 100 else "")
                    print(f"  [{msg['role'].upper()}] {snippet}")
                if not history:
                    print("  (none yet)")
                continue

            if user_input == "/summaries":
                summaries = chatbot.get_memories().get("summaries", [])
                print("\n📊 DETAILED SUMMARIES:")
                for i, s in enumerate(summaries, 1):
                    print(f"\n  Summary {i} ({s.get('turn_range','?')}):")
                    print(f"    Topics     : {', '.join(s.get('key_topics',[]))}")
                    print(f"    Decisions  : {', '.join(s.get('decisions_made',[])) or 'None'}")
                    print(f"    Action items: {', '.join(s.get('action_items',[])) or 'None'}")
                    if s.get("important_context"):
                        print(f"    Context    : {s['important_context']}")
                if not summaries:
                    print("  (none yet)")
                continue

            if user_input.startswith("/search "):
                query   = user_input[len("/search "):].strip()
                results = chatbot.search_memories(query, top_k=10)
                print(f"\n🔍 Results for: '{query}'")
                for i, r in enumerate(results, 1):
                    print(
                        f"  {i}. [{r.get('category','?')}] {r.get('text','')} "
                        f"(final={r.get('final_score',0):.3f}, "
                        f"salience={r.get('salience_score',0):.2f}, "
                        f"cosine={r.get('relevance_score',0):.3f}, "
                        f"recency={r.get('recency_score',0):.3f})"
                    )
                if not results:
                    print("  (no results)")
                continue

            if user_input == "/metrics":
                summary = chatbot.get_metrics()
                print("\n📊 OBSERVABILITY METRICS:")
                print("  Counts:")
                for k, v in sorted(summary.get("counts", {}).items()):
                    print(f"    {k}: {v}")
                print("  Sums:")
                for k, v in sorted(summary.get("sums", {}).items()):
                    print(f"    {k}: {v:.1f}")
                continue

            if user_input == "/prune":
                pruned = chatbot.prune_stale_memories()
                print(f"  ✅ Pruned {pruned} stale memories.")
                continue

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
            print("Try again or type 'bye' to exit.")


if __name__ == "__main__":
    main()
