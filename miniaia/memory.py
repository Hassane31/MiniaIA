"""Conversation memory with a character budget (a 3B model has a small context)."""
from __future__ import annotations


class ConversationMemory:
    def __init__(self, char_budget: int = 12000):
        self.char_budget = char_budget
        self.messages: list[dict] = []

    def add(self, role: str, content: str) -> None:
        if role not in ("user", "assistant"):
            raise ValueError(f"invalid role: {role}")
        self.messages.append({"role": role, "content": content})

    def clear(self) -> None:
        self.messages.clear()

    def __len__(self) -> int:
        return len(self.messages)

    def window(self) -> list[dict]:
        """Newest messages that fit the budget; always starts with a user message."""
        picked: list[dict] = []
        used = 0
        for msg in reversed(self.messages):
            size = len(msg["content"])
            if picked and used + size > self.char_budget:
                break
            picked.append(msg)
            used += size
        picked.reverse()
        while picked and picked[0]["role"] != "user":
            picked.pop(0)
        return [dict(m) for m in picked]
