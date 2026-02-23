"""
Mem0 Bridge — Shared Memory with Miya Android App
==================================================
"""

import os
import logging

logger = logging.getLogger(__name__)

MEM0_API_KEY = os.environ.get("MEM0_API_KEY", "")
mem0_client = None

if MEM0_API_KEY:
    try:
        from mem0 import MemoryClient
        mem0_client = MemoryClient(api_key=MEM0_API_KEY)
        logger.info("Mem0 connected - memories shared with Miya app")
    except Exception as e:
        logger.warning(f"Mem0 init failed: {e}")


def mem0_recall(query: str, user_id: str = "senthil") -> str:
    """Search Mem0 for relevant memories before responding."""
    if not mem0_client:
        return ""
    try:
        raw = mem0_client.search(
            query,
            filters={"user_id": user_id},
            top_k=5,
            threshold=0.5,
        )
        results = raw.get("results", []) if isinstance(raw, dict) else raw
        if not results:
            return ""
        memories = []
        for r in results:
            mem_text = r.get("memory", "")
            cat = r.get("categories") or []
            score = r.get("score", 0)
            if mem_text and score >= 0.5:
                tag = f" [{cat[0]}]" if cat else ""
                memories.append(f"- {mem_text}{tag}")
        if memories:
            return "\n[Miya's memories about this person]:\n" + "\n".join(memories)
        return ""
    except Exception as e:
        logger.debug(f"Mem0 recall error: {e}")
        return ""


def mem0_store(user_msg: str, assistant_msg: str, user_id: str = "senthil"):
    """Store conversation in Mem0 for long-term memory."""
    if not mem0_client:
        return
    try:
        if len(user_msg.strip()) < 5:
            return
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg[:500]}
        ]
        mem0_client.add(
            messages,
            user_id=user_id,
            metadata={"source": "miya_telegram"},
            infer=True,
        )
    except Exception as e:
        logger.debug(f"Mem0 store error: {e}")
