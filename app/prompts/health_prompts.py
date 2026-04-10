"""
Health-specific LLM prompt templates (Phase 2/4).
"""

CGM_ANALYSIS_PROMPT = """Analyze the following CGM data and provide insights.

CGM READINGS SUMMARY:
{readings_summary}

RECENT FOOD LOG:
{recent_food}

RECENT MOOD DATA:
{recent_mood}

RECENT ACTIVITY DATA:
{recent_activity}

TASK:
Analyze glucose patterns and provide:
1. Severity level (normal / mild / moderate / severe)
2. Likely causes for any spikes, hypos, or unusual patterns
3. Specific recommendations tailored to the observed patterns
4. Description of trends (rising, falling, stable, volatile)
5. Time-in-range assessment
6. Any notable correlations with food, mood, or activity

Be concise and actionable. Always include the medical disclaimer.
"""

CORRELATION_PROMPT = """Identify causal correlations between events in the user's timeline.

TIMELINE OF EVENTS (chronological order with timestamps and categories):
{timeline}

TASK:
Analyze the timeline and identify causal correlations. For each correlation found:
1. Describe the correlation insight clearly
2. Assign a confidence level: low, medium, or high
3. Provide an actionable recommendation

Consider:
- Food intake followed by glucose changes (30-120 min delay)
- Exercise followed by glucose drops
- Stress/mood events affecting glucose
- Medication timing and glucose response
- Sleep quality impacting next-day patterns

Output format:
- Correlation: [description]
- Confidence: [low/medium/high]
- Recommendation: [actionable advice]
"""

SYMPTOM_TRIAGE_PROMPT = """Assess user-reported symptoms and provide guidance.

SYMPTOMS REPORTED:
{symptoms}

CURRENT CGM DATA:
{cgm_data}

CURRENT MEDICATIONS:
{medications}

KNOWN HEALTH CONDITIONS:
{health_conditions}

TASK:
1. Assess severity: low, medium, high, or critical
2. Provide clear guidance text
3. Determine if physician escalation is needed (escalate: true/false)

IMPORTANT: You must ALWAYS include this medical disclaimer at the end:
"This is not medical advice. Always consult a qualified physician for medical decisions."

Escalation rules — escalate = true for:
- Chest pain or pressure
- CGM <54 mg/dL (severe hypoglycemia)
- CGM >300 mg/dL (severe hyperglycemia)
- Difficulty breathing
- Loss of consciousness or confusion
- Persistent vomiting
- Signs of DKA (fruity breath, extreme thirst, frequent urination, nausea)

Output format:
- Severity: [low/medium/high/critical]
- Guidance: [clear, actionable text]
- Escalate: [true/false]
- Disclaimer: [always included]
"""

NUTRITION_ADVICE_PROMPT = """Provide personalized nutrition guidance based on the user's food log and glucose response.

FOOD LOG:
{food_log}

CGM RESPONSE TO MEAL:
{cgm_response}

DIETARY MEMORIES (preferences, restrictions, goals):
{dietary_memories}

KNOWN HEALTH CONDITIONS:
{health_conditions}

TASK:
Provide personalized nutrition advice including:
1. Glycemic impact explanation — how the meal likely affected glucose
2. Portion recommendations for better glucose control
3. Healthier alternatives with similar satisfaction
4. Timing suggestions (e.g., post-meal walk to blunt spike)
5. Consider the user's known dietary preferences and restrictions

Be supportive and practical. Never shame or judge food choices.
Always include the medical disclaimer.
"""

NUDGE_GENERATION_PROMPT = """Generate a contextually appropriate proactive nudge for the user.

NUDGE TYPE:
{nudge_type}

CURRENT USER STATE:
{user_state}

RECENT CONTEXT:
{recent_context}

TASK:
Generate a nudge message that is:
1. Relevant to the user's current state
2. Supportive and non-judgmental
3. Actionable — suggest a specific next step
4. Personalized — reference known preferences or patterns

Output format:
- nudge_text: [max 120 characters — suitable for push notification]
- nudge_type: [glucose_alert / meal_reminder / medication_reminder / hydration / activity_reminder / mood_checkin / program_motivation]
- priority: [low / medium / high / critical]
"""

PROGRAM_GUIDANCE_PROMPT = """Provide personalized program guidance for today.

PROGRAM:
{program_name}

PROGRESS:
Current day: {current_day} of {total_days}
Completion: {progress_pct}%

RECENT HEALTH DATA:
{recent_health_data}

TASK:
Provide today's guidance including:
1. Today's tasks and activities from the program
2. Personalization based on recent health data (CGM trends, mood, activity)
3. Motivational message referencing milestones achieved
4. Any adjustments or cautions based on current health state

Be encouraging and specific. Celebrate progress.
"""
