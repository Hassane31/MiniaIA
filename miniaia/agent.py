"""The agent loop:  UNDERSTAND -> PLAN -> EXECUTE -> OBSERVE -> ANALYZE -> ACT

The LLM only *proposes* one JSON decision per step:
    answer       -> final reply
    use_tool     -> call a built-in tool (validated + executed by Python)
    run_command  -> propose ONE shell command (security.py + approval + executor)
It never executes anything itself. Retries are never special: a corrected command goes
through exactly the same validation and approval as the first attempt.
"""
from __future__ import annotations

import getpass
import json
from contextlib import contextmanager
from typing import Optional

from .command_executor import CommandExecutor, ExecResult
from .config import Config
from .decision import Decision, DecisionError, parse_decision          # noqa: F401  (re-exported)
from .memory import ConversationMemory
from .ollama_client import OllamaClient
from .tools import ToolRegistry, ToolResult

FAILED_STATUSES = frozenset({"failed", "timeout", "error", "invalid"})

_PROMPT_BASE = """You are miniaIA, a careful Linux assistant on the user's Kali Linux laptop (user: <<USER>>, cwd: <<CWD>>).
Reply with ONE JSON object and nothing else. Fields: thought (one short sentence), action, tool, args, command, answer, recommendation.

Actions:
- "answer": final reply in "answer" (Markdown allowed); optional "recommendation" = one practical next step.
<<ACTIONS>>
Rules:
- Facts about this computer come ONLY from TOOL RESULT / OBSERVATION text. Quote numbers exactly. Never invent data.
- Earlier results may be outdated: call the tool again to get current values.
- One step at a time: call one tool or command, read the result, then continue or answer.
- To create or delete files, folders or user accounts ALWAYS use the action tools (create_folder, create_file, delete_path, create_user, delete_user): they ask the human and verify the result. Never use mkdir, touch, rm, useradd or userdel commands for that.
- A command must be written exactly as typed in a terminal: do NOT put quotes or backticks around the whole command. Commands run WITHOUT a shell: simple commands, plain arguments and the pipe | only; never ; && || > < $VAR $(...). Avoid awk/sed when a tool exists.
- If something fails you get the error and hints: explain briefly, then propose ONE corrected command or another tool. A safety layer may block a command or ask the user; if refused, do not retry it.
- Never try to disable security tools or read secrets. Use sudo only when really necessary.
- Reply in the language the user writes in (English, French or Arabic). Be clear and simple.
- Greetings and general knowledge: just answer.
"""
_ACTIONS_TOOLS = """- "use_tool": call a built-in tool (preferred for questions about this computer, its network, or a file). Set "tool" and "args".
- "run_command": run ONE Linux command only when no tool fits. Set "command".

TOOLS (they measure the real system):
<<TOOLS>>
"""
_ACTIONS_NO_TOOLS = '- "run_command": run ONE Linux command. Set "command". Its output comes back as an OBSERVATION.\n'
_EXAMPLES = """
Examples:
user: why is my laptop slow? -> {"thought":"measure the system first","action":"use_tool","tool":"system_overview","args":{}}
user: what is using port 8080? -> {"thought":"inspect that port","action":"use_tool","tool":"port_info","args":{"port":8080}}
user: is ssh running? -> {"thought":"check the service","action":"use_tool","tool":"service_status","args":{"service":"ssh"}}
user: create a folder named aziz on the desktop -> {"thought":"use the action tool","action":"use_tool","tool":"create_folder","args":{"path":"~/Desktop/aziz"}}
user: delete the file notes.txt -> {"thought":"move it to the trash","action":"use_tool","tool":"delete_path","args":{"path":"notes.txt"}}
user: create a user called sara -> {"thought":"use the action tool","action":"use_tool","tool":"create_user","args":{"username":"sara"}}
user: list python files here -> {"thought":"simple listing","action":"run_command","command":"ls *.py"}
user: hello -> {"thought":"greeting","action":"answer","answer":"Hello! How can I help with your system?"}
"""


class AgentUI:
    """Hooks the agent calls to talk to the human. Default: do nothing (used in tests)."""

    @contextmanager
    def thinking(self, phase: str = "thinking"):
        yield lambda chunk: None

    @contextmanager
    def activity(self, label: str):
        yield

    def show_command(self, command: str) -> None: ...
    def show_result(self, result: ExecResult) -> None: ...
    def show_tool_call(self, name: str, args: dict) -> None: ...
    def show_tool_result(self, result: ToolResult) -> None: ...
    def show_answer(self, text: str) -> None: ...
    def show_recommendation(self, text: str) -> None: ...
    def show_notice(self, text: str) -> None: ...


def build_schema(tools: Optional[ToolRegistry]) -> dict:
    """JSON schema handed to Ollama so decoding is constrained to a valid decision."""
    actions = ["answer", "run_command"] + (["use_tool"] if tools else [])
    props: dict = {"thought": {"type": "string"}, "action": {"type": "string", "enum": actions},
                   "command": {"type": "string"}, "answer": {"type": "string"}, "recommendation": {"type": "string"}}
    if tools:
        props["tool"] = {"type": "string", "enum": tools.names()}
        props["args"] = tools.args_schema()
    return {"type": "object", "properties": props, "required": ["thought", "action"]}


class Agent:
    def __init__(self, config: Config, client: OllamaClient, executor: CommandExecutor,
                 memory: ConversationMemory, ui: Optional[AgentUI] = None, tools: Optional[ToolRegistry] = None):
        self.config, self.client, self.executor, self.memory = config, client, executor, memory
        self.ui = ui or AgentUI()
        self.tools = tools

    # ------------------------------------------------------------------ prompt
    def _system_prompt(self) -> dict:
        try:
            user = getpass.getuser()
        except Exception:
            user = "user"
        actions = _ACTIONS_TOOLS.replace("<<TOOLS>>", self.tools.prompt_lines()) if self.tools else _ACTIONS_NO_TOOLS
        text = (_PROMPT_BASE.replace("<<ACTIONS>>", actions).replace("<<USER>>", user)
                .replace("<<CWD>>", self.executor.cwd) + _EXAMPLES)
        return {"role": "system", "content": text}

    # ------------------------------------------------------------------ one model call
    def _decide(self, phase: str = "thinking") -> Decision:
        valid = set(self.tools.names()) if self.tools else set()
        messages = [self._system_prompt()] + self.memory.window()
        schema = build_schema(self.tools)
        with self.ui.thinking(phase) as on_token:
            raw = self.client.chat(messages, json_format=schema, on_token=on_token)
        try:
            return parse_decision(raw, valid)
        except DecisionError as first:
            retry = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Invalid reply ({first}). Answer again with ONE valid JSON object only."},
            ]
            with self.ui.thinking("thinking") as on_token:
                raw2 = self.client.chat(retry, json_format=schema, on_token=on_token)
            try:
                return parse_decision(raw2, valid)
            except DecisionError:
                text = (raw2 or raw).strip()
                return Decision("answer", answer=text or "(the model returned an empty reply)")

    # ------------------------------------------------------------------ the loop
    def run_turn(self, user_text: str) -> str:
        """Handle one user message; returns the final answer text."""
        self.memory.add("user", user_text)
        seen: set = set()
        failures = 0
        phase = "thinking"

        for _ in range(self.config.max_agent_steps):
            decision = self._decide(phase)
            if decision.action == "answer":
                return self._finish(decision.answer, decision.recommendation)

            self.memory.add("assistant", self._record(decision))
            if decision.action == "use_tool":
                observation, failed = self._step_tool(decision, seen)
            else:
                observation, failed = self._step_command(decision, seen)
            self.memory.add("user", observation)
            failures = failures + 1 if failed else 0
            phase = "analyzing"
            if failures >= self.config.max_consecutive_failures:
                return self._force_final("Too many consecutive failures; summarising.",
                                         "Several attempts in a row failed. Reply NOW with action \"answer\": explain what "
                                         "went wrong and what the user could try. Do not propose another command.")

        return self._force_final(f"Step limit ({self.config.max_agent_steps}) reached; summarising.",
                                 "Step limit reached. Reply NOW with action \"answer\": summarise what you found "
                                 "and what remains unknown.")

    # ------------------------------------------------------------------ steps
    @staticmethod
    def _record(d: Decision) -> str:
        if d.action == "use_tool":
            return json.dumps({"thought": d.thought, "action": "use_tool", "tool": d.tool, "args": d.args}, ensure_ascii=False)
        return json.dumps({"thought": d.thought, "action": "run_command", "command": d.command}, ensure_ascii=False)

    def _step_tool(self, d: Decision, seen: set) -> tuple:
        label = f"{d.tool}({', '.join(f'{k}={v}' for k, v in d.args.items())})"
        self.ui.show_tool_call(d.tool, d.args)
        key = ("tool", d.tool, json.dumps(d.args, sort_keys=True, default=str))
        if key in seen:
            return ("TOOL RESULT [skipped]: you already called this tool with the same arguments in this request. "
                    "Use that result to answer, or call something different."), False
        seen.add(key)
        with self.ui.activity(f"Calling tool {label}"):
            result = self.tools.call(d.tool, d.args)
        self.ui.show_tool_result(result)
        return result.observation(self.config.max_output_chars), result.status in ("error", "refused")

    def _step_command(self, d: Decision, seen: set) -> tuple:
        self.ui.show_command(d.command)
        if d.command in seen:
            return ("OBSERVATION [skipped]: you already ran this exact command. Do not repeat it; "
                    "use what you learned to answer, or try something different."), True
        seen.add(d.command)
        result = self.executor.run(d.command)
        self.ui.show_result(result)
        return result.observation(self.config.max_output_chars), result.status in FAILED_STATUSES

    # ------------------------------------------------------------------ finishing
    def _finish(self, answer: str, recommendation: str = "") -> str:
        self.memory.add("assistant", json.dumps({"action": "answer", "answer": answer}, ensure_ascii=False))
        self.ui.show_answer(answer)
        if recommendation:
            self.ui.show_recommendation(recommendation)
        return answer

    def _force_final(self, notice: str, instruction: str) -> str:
        self.ui.show_notice(notice)
        self.memory.add("user", instruction)
        d = self._decide("analyzing")
        text = d.answer or d.thought or "I could not finish this request."
        return self._finish(text, d.recommendation)
