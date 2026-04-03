"""
Conflict resolution: detect when a new candidate contradicts an existing memory
and decide whether to supersede, keep both, or ignore the new candidate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.models.schemas import ConflictRecord, ConflictResolutionResult, MemoryCandidate
from app.observability.metrics import metrics
from app.prompts.templates import CONFLICT_RESOLUTION_PROMPT


class ConflictResolver:
    def __init__(self, memory_llm) -> None:
        self._resolver = memory_llm.with_structured_output(ConflictResolutionResult)

    def resolve(
        self,
        candidates: List[MemoryCandidate],
        existing_memories: List[Dict[str, Any]],
    ) -> Tuple[List[MemoryCandidate], Dict[int, str]]:
        """
        Returns:
          - filtered_candidates: candidates that should be written (ignore_new removed)
          - supersedes_map: {candidate_index -> existing_memory_id} for supersede actions
        """
        if not candidates or not existing_memories:
            return candidates, {}

        candidates_text = "\n".join(
            f"{i}. [{c.category}] {c.text}" for i, c in enumerate(candidates)
        )
        existing_text = "\n".join(
            f"id={m['id']} [{m.get('category','?')}] {m.get('text','')}"
            for m in existing_memories
        )

        try:
            result: ConflictResolutionResult = self._resolver.invoke(
                [
                    SystemMessage(
                        content=CONFLICT_RESOLUTION_PROMPT.format(
                            candidates=candidates_text,
                            existing_memories=existing_text,
                        )
                    ),
                    HumanMessage(content="Detect and resolve conflicts."),
                ]
            )
        except Exception as e:
            print(f"⚠️  Conflict resolution failed: {e}")
            return candidates, {}

        ignored_indices: set[int]    = set()
        supersedes_map:  Dict[int, str] = {}

        for conflict in result.conflicts:
            idx = conflict.new_candidate_idx
            if conflict.action == "ignore_new":
                ignored_indices.add(idx)
            elif conflict.action == "supersede":
                supersedes_map[idx] = conflict.existing_memory_id
            # "keep_both" → no special action needed

        filtered = [
            c for i, c in enumerate(candidates) if i not in ignored_indices
        ]

        metrics.log(
            "conflict_resolution",
            total=len(candidates),
            ignored=len(ignored_indices),
            superseded=len(supersedes_map),
        )
        return filtered, supersedes_map
