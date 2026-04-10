# """
# setup_structure.py
# ------------------
# Run this script from INSIDE the app/ folder:

#     cd "C:\\Users\\Reyna\\OneDrive - Reyna Solutions LLP\\Desktop\\Personal AI assistant\\app"
#     python setup_structure.py

# It will:
#   1. Create all subfolders
#   2. Move files into the correct subfolders
#   3. Create all __init__.py files
#   4. Move main.py and .env up one level (to Personal AI assistant/)
#   5. Print a final confirmation
# """

# import os
# import shutil
# from pathlib import Path

# # ── Detect where we are ───────────────────────────────────────────────────────
# HERE        = Path(__file__).parent.resolve()   # app/ folder
# PARENT      = HERE.parent                        # Personal AI assistant/ folder

# print(f"\n📂 Working inside : {HERE}")
# print(f"📂 Parent folder  : {PARENT}\n")

# # ── File → destination subfolder mapping ─────────────────────────────────────
# MOVES = {
#     "settings.py":     "config",
#     "pinecone_ltm.py": "memory",
#     "controller.py":   "memory",
#     "conflict.py":     "memory",
#     "builder.py":      "graph",
#     "nodes.py":        "graph",
#     "schemas.py":      "models",
#     "metrics.py":      "observability",
#     "isolation.py":    "security",
#     "templates.py":    "prompts",
# }

# # Files that go to the PARENT folder (one level up)
# MOVE_TO_PARENT = ["main.py", ".env"]

# # All subfolders that need to exist
# SUBFOLDERS = [
#     "config",
#     "memory",
#     "graph",
#     "models",
#     "observability",
#     "security",
#     "prompts",
# ]

# # ── Step 1: Create subfolders ─────────────────────────────────────────────────
# print("📁 Step 1: Creating subfolders ...")
# for folder in SUBFOLDERS:
#     path = HERE / folder
#     path.mkdir(exist_ok=True)
#     print(f"   ✅ {folder}/")

# # ── Step 2: Move files into subfolders ────────────────────────────────────────
# print("\n📦 Step 2: Moving files into subfolders ...")
# for filename, subfolder in MOVES.items():
#     src  = HERE / filename
#     dest = HERE / subfolder / filename
#     if src.exists():
#         if dest.exists():
#             print(f"   ⏭️  Skipped (already there): {subfolder}/{filename}")
#         else:
#             shutil.move(str(src), str(dest))
#             print(f"   ✅ {filename}  →  {subfolder}/")
#     else:
#         print(f"   ⚠️  Not found (skipping): {filename}")

# # ── Step 3: Create __init__.py files ─────────────────────────────────────────
# print("\n📝 Step 3: Creating __init__.py files ...")

# init_locations = [HERE] + [HERE / sf for sf in SUBFOLDERS]
# for loc in init_locations:
#     init_file = loc / "__init__.py"
#     if not init_file.exists():
#         init_file.write_text("# auto-generated\n")
#         rel = init_file.relative_to(PARENT)
#         print(f"   ✅ {rel}")
#     else:
#         rel = init_file.relative_to(PARENT)
#         print(f"   ⏭️  Exists: {rel}")

# # ── Step 4: Move main.py and .env to parent ───────────────────────────────────
# print("\n🚚 Step 4: Moving main.py and .env to parent folder ...")
# for filename in MOVE_TO_PARENT:
#     src  = HERE / filename
#     dest = PARENT / filename
#     if src.exists():
#         if dest.exists():
#             print(f"   ⏭️  Skipped (already in parent): {filename}")
#         else:
#             shutil.move(str(src), str(dest))
#             print(f"   ✅ {filename}  →  {PARENT}/")
#     else:
#         print(f"   ⚠️  Not found (skipping): {filename}")

# # ── Step 5: Verify final structure ───────────────────────────────────────────
# print("\n🔍 Step 5: Final structure check ...")

# EXPECTED = {
#     PARENT / "main.py":                           "main.py",
#     PARENT / ".env":                              ".env",
#     HERE   / "__init__.py":                       "app/__init__.py",
#     HERE   / "chatbot.py":                        "app/chatbot.py",
#     HERE   / "cli.py":                            "app/cli.py",
#     HERE   / "config"   / "settings.py":          "app/config/settings.py",
#     HERE   / "memory"   / "pinecone_ltm.py":      "app/memory/pinecone_ltm.py",
#     HERE   / "memory"   / "controller.py":        "app/memory/controller.py",
#     HERE   / "memory"   / "conflict.py":          "app/memory/conflict.py",
#     HERE   / "graph"    / "builder.py":           "app/graph/builder.py",
#     HERE   / "graph"    / "nodes.py":             "app/graph/nodes.py",
#     HERE   / "models"   / "schemas.py":           "app/models/schemas.py",
#     HERE   / "observability" / "metrics.py":      "app/observability/metrics.py",
#     HERE   / "security" / "isolation.py":         "app/security/isolation.py",
#     HERE   / "prompts"  / "templates.py":         "app/prompts/templates.py",
# }

# all_ok = True
# for path, label in EXPECTED.items():
#     if path.exists():
#         print(f"   ✅ {label}")
#     else:
#         print(f"   ❌ MISSING: {label}")
#         all_ok = False

# # ── Done ──────────────────────────────────────────────────────────────────────
# print("\n" + "=" * 60)
# if all_ok:
#     print("🎉 Setup complete! Now run:\n")
#     print(f'   cd "{PARENT}"')
#     print("   python main.py\n")
# else:
#     print("⚠️  Some files are missing — check the ❌ items above.")
#     print("   Make sure all 13 files are inside the app/ folder first.\n")
# print("=" * 60)




import os

# Base directory
base_dir = r"C:\Users\Reyna\OneDrive - Reyna Solutions LLP\Desktop\Personal AI assistant"

# Folder structure and files with comments
structure = {
    "app/api": {
        "__init__.py": "# API package",
        "main.py": "# FastAPI app + CORS + lifespan",
        "dependencies.py": "# Dependency Injection: get_chatbot, get_current_user",
        "middleware": {
            "auth.py": "# JWT validation",
            "rate_limit.py": "# Rate limiting middleware"
        },
        "schemas": {
            "chat.py": "# Chat request/response schemas",
            "health.py": "# CGM, mood, food, activity schemas",
            "memories.py": "# Memory CRUD schemas",
            "programs.py": "# Program guidance schemas",
            "system.py": "# Health, metrics, prune schemas"
        },
        "routes": {
            "chat.py": "# POST /api/chat",
            "cgm.py": "# POST /api/cgm/readings (Phase 3)",
            "mood.py": "# POST /api/mood (Phase 3)",
            "food.py": "# POST /api/food (Phase 3)",
            "activity.py": "# POST /api/activity (Phase 3)",
            "memories.py": "# GET/DELETE /api/memories",
            "programs.py": "# GET/POST /api/programs (Phase 6)",
            "nudges.py": "# GET /api/nudges (Phase 5)",
            "system.py": "# /health, /metrics, /prune"
        }
    },
    "app/chatbot.py": "# Chatbot module extended with health context",
    "app/cli.py": "# CLI tool (unchanged)",
    "app/config": {
        "settings.py": "# Configuration settings extended with health settings"
    },
    "app/models": {
        "schemas.py": "# Extended schemas with health models",
        "health_schemas.py": "# CGM, mood, food, activity models (Phase 3)"
    },
    "app/prompts": {
        "templates.py": "# Health-aware prompt templates (Phase 2)",
        "health_prompts.py": "# CGM analysis and triage prompts (Phase 4)"
    },
    "app/memory": {
        "pinecone_ltm.py": "# Memory persistence (unchanged)",
        "controller.py": "# Memory controller extended with health data methods",
        "conflict.py": "# Conflict resolution (unchanged)"
    },
    "app/health": {
        "__init__.py": "# Health domain package",
        "cgm_processor.py": "# CGM data normalization + pattern detection",
        "correlation_engine.py": "# Cross-signal correlation engine",
        "nudge_engine.py": "# Proactive nudge generation",
        "food_recognition.py": "# Image-based food detection wrapper",
        "program_tracker.py": "# Program progress and task management",
        "symptom_triage.py": "# Rule-based symptom assessment",
        "knowledge_directory.py": "# Knowledge base integration"
    },
    "app/graph": {
        "builder.py": "# Graph builder extended with health nodes",
        "nodes.py": "# Graph nodes extended with health processing"
    },
    "app/integrations": {
        "firebase_push.py": "# FCM push notifications",
        "speech.py": "# Speech-to-text integration",
        "nutrition_api.py": "# Nutritionix/Edamam API integration"
    },
    "app/observability": {
        "metrics.py": "# Metrics collection (unchanged)"
    },
    "app/security": {
        "isolation.py": "# Security isolation (unchanged)",
        "auth.py": "# JWT + RBAC authentication"
    },
    "app/test_suite.py": "# Test suite extended with health tests"
}

def create_structure(base, struct):
    for name, content in struct.items():
        path = os.path.join(base, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content + "\n")
                print(f"Created {path}")
            else:
                print(f"{path} already exists")

# Run the creation
create_structure(base_dir, structure)