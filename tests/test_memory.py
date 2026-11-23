import pytest

from miniaia.memory import ConversationMemory


def test_add_and_clear():
    m = ConversationMemory(5000)
    m.add("user", "hi")
    m.add("assistant", "yo")
    assert len(m) == 2
    m.clear()
    assert len(m) == 0


def test_invalid_role():
    with pytest.raises(ValueError):
        ConversationMemory().add("system", "x")


def test_window_trims_oldest_and_starts_with_user():
    m = ConversationMemory(1000)
    for i in range(10):
        m.add("user", f"u{i}" + "x" * 200)
        m.add("assistant", f"a{i}" + "y" * 200)
    w = m.window()
    assert w[0]["role"] == "user"
    assert sum(len(x["content"]) for x in w) <= 1000
    assert w[-1]["content"].startswith("a9")


def test_window_keeps_last_message_even_if_huge():
    m = ConversationMemory(1000)
    m.add("user", "z" * 5000)
    assert len(m.window()) == 1


def test_window_returns_copies():
    m = ConversationMemory(1000)
    m.add("user", "hi")
    m.window()[0]["content"] = "changed"
    assert m.messages[0]["content"] == "hi"
