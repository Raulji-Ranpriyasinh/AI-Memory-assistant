"""
All LLM prompt templates in one place.
"""

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

CONFLICT_RESOLUTION_PROMPT = """Detect conflicts between new memory candidates and existing memories.

NEW CANDIDATES (indexed 0-based):
{candidates}

EXISTING MEMORIES:
{existing_memories}

For each new candidate, check if it CONTRADICTS an existing memory.
If a conflict is found:
- "supersede": new info replaces old (e.g., user moved city, changed job)
- "keep_both": both valid at once (e.g., likes pizza AND sushi)
- "ignore_new": existing is more reliable or new is noise

Return a list of conflicts. If no conflict, return an empty list.
Each entry must include: new_candidate_idx, existing_memory_id, action, reason.
"""
