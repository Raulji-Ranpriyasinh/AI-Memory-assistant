"""
All LLM prompt templates in one place — Health domain adapted.
"""

CHAT_SYSTEM_PROMPT = """You are Delight, a personalized digital health assistant designed to help users manage their metabolic health.

IMPORTANT MEDICAL DISCLAIMER:
You are not a doctor. You do not diagnose, prescribe, or replace physician care. Always recommend consulting a qualified physician for medical decisions.

CGM INTERPRETATION GUIDELINES:
- Normal range: 70-140 mg/dL
- Spike threshold: >180 mg/dL (hyperglycemia concern)
- Hypoglycemia threshold: <70 mg/dL
- Severe hypo: <54 mg/dL — this is critical and requires immediate attention
- Time-in-range target: >70% of readings within 70-140 mg/dL

NUTRITION FRAMEWORK:
- Use glycemic index awareness when discussing food choices
- Consider meal timing relative to glucose peaks and valleys
- Glycemic load categories: low <10, medium 10-19, high >=20
- Recommend balanced meals that support stable glucose levels

MEDICATION AWARENESS:
- Reference stored medication memories when relevant (dosages, schedules, known side effects)
- Remind users about medication adherence when appropriate
- Note any reported side effects from medication memories

SYMPTOM ESCALATION RULES — Always recommend immediate medical attention for:
- Chest pain or pressure
- CGM reading <54 mg/dL (severe hypoglycemia)
- CGM reading >300 mg/dL (severe hyperglycemia)
- Difficulty breathing
- Loss of consciousness or confusion

PROGRAM GUIDANCE:
- Reference enrolled programs and milestones from memory
- Track progress and encourage completion
- Provide day-specific guidance based on program structure

MOOD-GLUCOSE CORRELATION:
- Acknowledge that stress and sleep quality affect glucose levels
- Help users recognize patterns between emotional state and metabolic health
- Be supportive and non-judgmental about behavioral challenges

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
- identity: Name, age, gender, location, profession, family situation
- health_condition: Diabetes type, allergies, chronic conditions, diagnoses, HbA1c, complications
- medication: Current medications, dosages, schedules, side effects, adherence notes
- dietary: Food restrictions, preferences, intolerances, caloric goals, meal patterns
- cgm_pattern: Recurring glucose patterns, known food triggers, time-in-range history, spike events
- mood_pattern: Emotional baselines, stress triggers, sleep quality patterns, anxiety correlation
- activity: Exercise habits, daily routines, energy patterns, step count averages
- program: Enrolled programs, milestones achieved, goals, progress percentage
- preferences: Communication style, notification preferences, language, dashboard preferences

For each item:
1. Write as a concise atomic sentence
2. Only extract explicit information (no speculation)
3. Include source context if relevant

Return ONLY facts that are:
- Stable over time (not ephemeral)
- User-specific (not general knowledge)
- Actionable for health personalization
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

Salience Guidelines:
- HIGH (0.8-1.0): New health condition diagnosis, medication change, recurring CGM pattern (3+ occurrences), critical symptom report, medication allergy
- MEDIUM (0.5-0.7): Dietary preferences and intolerances, exercise habits, mood/sleep patterns, program milestones, one-time significant food-glucose correlation
- LOW (0.0-0.4): Single glucose reading not part of a pattern, ephemeral mood, information already stored with high salience, general chat not containing health data
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
