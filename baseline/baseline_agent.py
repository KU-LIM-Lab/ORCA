"""
baseline_agent.py

Baseline Agent for User Study (3-Step Runner)

Design goals:
- Step-level session modes (Step 1 -> Step 2 -> Step 3).
- Inside each step: free-form chat + tool-calling loop.
- Step transition: user enters a fixed command (e.g., /next) -> agent finalizes the step
  by saving required artifacts via tool calls -> then moves to next step.
- Step-to-step continuity is ensured by a persisted state (state.json) + artifact paths,
  not by long chat history.

Expected existing modules (you already have most):
- baseline.prompts.system.load_baseline_system_prompt()  # general baseline system prompt
- baseline.tools: get_schema, run_sql, run_python, save_artifact, final_answer,
                  set_tool_context, BASELINE_TOOLS

Notes:
- This file DOES NOT require LangGraph.
- You can add more commands later (/help, /retry, etc.).
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from baseline.prompts.system import load_baseline_system_prompt
from baseline.tools import (
    set_tool_context,
    BASELINE_TOOLS,
)

logger = logging.getLogger(__name__)

BASELINE_SYSTEM_PROMPT = load_baseline_system_prompt()

# -----------------------------
# Tool schemas (OpenAI function calling)
# -----------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": "Retrieve database schema including tables, columns, types, and relationships. Use this at the start to understand DB structure.",
            "parameters": {
                "type": "object",
                "properties": {"db_id": {"type": "string"}},
                "required": ["db_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute SQL query and return results. Result DataFrame is stored for Python as 'df'.",
            "parameters": {
                "type": "object",
                "properties": {"db_id": {"type": "string"}, "sql": {"type": "string"}},
                "required": ["db_id", "sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code with access to 'df' (last SQL result) and any cached dataframes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "context_vars": {"type": "object", "additionalProperties": True},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_artifact",
            "description": "Save an artifact. Step1 needs sql+dataset; Step2 needs graph or graph_adj; Step3 needs ate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artifact_type": {
                        "type": "string",
                        "enum": ["sql", "dataset", "graph", "graph_adj", "ate", "estimation_spec", "schema"],
                    },
                    "data_ref": {"type": "string"},
                    "step_id": {"type": "string", "enum": ["1", "2", "3"]},
                    "filename": {"type": "string"},
                },
                "required": ["artifact_type", "step_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Signal completion after all steps are complete.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}, "all_steps_complete": {"type": "boolean"}},
                "required": ["summary", "all_steps_complete"],
            },
        },
    },
]

# -----------------------------
# Commands
# -----------------------------
CMD_NEXT = "/next"
CMD_FINAL = "/final"
CMD_HELP = "/help"
CMD_QUIT = "/quit"

ALIASES_NEXT = {"next"}
ALIASES_FINAL = {"final"}
ALIASES_QUIT = {"quit", "bye", "exit", "goodbye", "종료"}

# -----------------------------
# Step spec
# -----------------------------
STEP_NAMES = {
    "1": "Data Exploration & Extraction",
    "2": "Causal Graph Discovery",
    "3": "Causal Effect Estimation",
}

STEP_REQUIRED_ARTIFACTS = {
    "1": {"sql", "dataset"},
    "2": {"graph", "graph_adj"},  # either one is fine (special handling)
    "3": {"ate"},
}


@dataclass
class BaselineState:
    """Persisted session state (do NOT rely on LLM memory alone)."""

    participant_id: str
    session_id: str
    db_id: str

    # Current step
    current_step: str = "1"

    # Artifacts saved (by step)
    artifacts_by_step: Dict[str, List[Dict[str, Any]]] = None

    # Useful carry-over metadata
    user_question: Optional[str] = None
    treatment: Optional[str] = None
    outcome: Optional[str] = None
    confounders: Optional[List[str]] = None
    
    # Last SQL query executed (for tracking)
    last_sql_query: Optional[str] = None

    # Convenience: last-known final artifact paths
    step1_sql_path: Optional[str] = None
    step1_dataset_path: Optional[str] = None
    step2_graph_path: Optional[str] = None
    step2_graph_adj_path: Optional[str] = None
    step3_ate_path: Optional[str] = None

    def __post_init__(self):
        if self.artifacts_by_step is None:
            self.artifacts_by_step = {"1": [], "2": [], "3": []}
        if self.confounders is None:
            self.confounders = []


class BaselineAgent:
    """
    3-step runner baseline agent.

    - Each step has its own short message context.
    - State is persisted to disk.
    - Step finalize happens on user command (/next).
    """

    def __init__(
        self,
        participant_id: str,
        session_id: str,
        db_id: str,
        run_context: Any,
        model: str = "gpt-4o-mini",
        max_step_turns: int = 40,
        max_tool_depth: int = 20,
        history_keep_turns: int = 12,
    ):
        self.participant_id = participant_id
        self.session_id = session_id
        self.db_id = db_id
        self.run_context = run_context
        self.model = model

        # limits
        self.max_step_turns = max_step_turns
        self.max_tool_depth = max_tool_depth
        self.history_keep_turns = history_keep_turns

        # tracking
        self.client = OpenAI()
        self.event_logger = run_context.get_event_logger()
        self.artifact_manager = run_context.get_artifact_manager()

        # tool context (make sure tools can log against correct step)
        set_tool_context(
            event_logger=self.event_logger,
            artifact_manager=self.artifact_manager,
            current_step="1"
        )

        # persisted state path
        self.state_path = Path(run_context.artifacts_dir) / "state.json"
        self.state = self._load_or_init_state()

        # Step-level message buffer (reset on step entry)
        self.messages: List[Dict[str, Any]] = []
        self._enter_step(self.state.current_step, fresh=True)

        self.analysis_complete = False
        
        # Flag to skip trim during finalization
        self._in_finalization = False

    # -----------------------------
    # State IO
    # -----------------------------
    def _load_or_init_state(self) -> BaselineState:
        if self.state_path.exists():
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            st = BaselineState(**raw)
            return st
        st = BaselineState(
            participant_id=self.participant_id,
            session_id=self.session_id,
            db_id=self.db_id,
        )
        self._save_state(st)
        return st

    def _save_state(self, st: Optional[BaselineState] = None) -> None:
        st = st or self.state
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(asdict(st), ensure_ascii=False, indent=2), encoding="utf-8")

    # -----------------------------
    # CLI loop
    # -----------------------------
    def run_interactive(self) -> None:
        print("\n" + "=" * 70)
        print("🤖 Baseline Agent")
        print("=" * 70)
        print(f"Participant: {self.participant_id} | Session: {self.session_id} | DB: {self.db_id}")
        print("Commands: /next (finalize step), /final (finish), /help, /quit")
        print("=" * 70)

        if self.event_logger:
            self.event_logger.log_session_start(
                metadata={
                    "participant_id": self.participant_id,
                    "session_id": self.session_id,
                    "db_id": self.db_id,
                    "model": self.model,
                    "mode": "baseline_agent",
                }
            )

        # Step loop (the user can also quit early)
        while not self.analysis_complete:
            step_id = self.state.current_step
            step_name = STEP_NAMES[step_id]
            print(f"\n📌 Current Step {step_id}: {step_name}")

            # run step chat loop
            ended_by = self._run_step_chat_loop(step_id)

            if ended_by == "quit":
                break
            if ended_by == "final":
                break
            if ended_by == "next":
                # finalize + transition
                ok = self._finalize_step(step_id)
                if not ok:
                    print("\n⚠️ Step finalize failed. Continue in the same step and try again.")
                    continue

                if step_id == "3":
                    # All steps done -> allow /final to finish
                    print("\n✅ Step 3 finalized. Type /final to finish and write the final summary.")
                    
                    # Clean up messages to prepare for /final command
                    # Keep system messages and add a clear instruction for final step
                    sys_msgs = [m for m in self.messages if m["role"] == "system"]
                    self.messages = sys_msgs
                    self.messages.append({
                        "role": "system",
                        "content": (
                            "Step 3 is complete. All required artifacts have been saved.\n"
                            "When the user types /final, provide a concise summary and call final_answer()."
                        )
                    })
                    
                    # stay in step 3; user must /final to end (better experimental control)
                    continue

                # transition to next step
                next_step = str(int(step_id) + 1)
                self.state.current_step = next_step
                self._save_state()

                self._enter_step(next_step, fresh=True)
                continue

        self._finalize_session()

    def _run_step_chat_loop(self, step_id: str) -> str:
        """
        Returns: "next" | "final" | "quit"
        """
        turn = 0
        while turn < self.max_step_turns:
            user_input = input("\n🧑 You: ").strip()
            if not user_input:
                continue

            cmd = self._parse_command(user_input)
            if cmd == "help":
                self._print_help()
                continue
            if cmd == "quit":
                return "quit"
            if cmd == "next":
                return "next"
            if cmd == "final":
                # Only allow final when step 3 is finalized (ate saved)
                if step_id != "3":
                    print("⚠️ /final is only allowed in Step 3.")
                    if self.event_logger:
                        self.event_logger.log_event(
                            event_type="command_rejected",
                            metadata={
                                "command": "/final",
                                "reason": "not_in_step_3",
                                "current_step": step_id,
                            }
                        )
                    continue
                if not self._is_step_satisfied("3"):
                    print("⚠️ Step 3 is not finalized yet (ATE artifact missing). Use /next first.")
                    if self.event_logger:
                        self.event_logger.log_event(
                            event_type="command_rejected",
                            metadata={
                                "command": "/final",
                                "reason": "step3_not_finalized",
                                "current_step": step_id,
                            }
                        )
                    continue
                # call finalization summary turn
                self._final_answer()
                return "final"

            # Normal chat turn
            self._append_user(user_input)
            self._tool_chat_round(step_id=step_id)
            turn += 1

            # rolling history for step chat (keep it small)
            self._trim_history()

        print(f"\n⚠️ Reached max turns for this step ({self.max_step_turns}). Use /next or /quit.")
        return "next"  # force progress to avoid infinite sessions

    # -----------------------------
    # Step lifecycle
    # -----------------------------
    def _enter_step(self, step_id: str, fresh: bool = True) -> None:
        """
        Build step-specific context using:
        - base system prompt
        - step instruction header
        - persisted state summary (artifact paths etc.)
        """
        if fresh:
            self.messages = [{"role": "system", "content": BASELINE_SYSTEM_PROMPT}]
            self.messages.append({"role": "system", "content": self._step_header(step_id)})
            self.messages.append({"role": "system", "content": self._state_summary_for_step(step_id)})

        # update tools' current step for logging
        set_tool_context(
            event_logger=self.event_logger,
            artifact_manager=self.artifact_manager,
            current_step=step_id
        )

        if self.event_logger:
            self.event_logger.log_step_enter(
                step_id=step_id,
                substep="chat",
                metadata={"step_name": STEP_NAMES.get(step_id)},
            )

    def _finalize_step(self, step_id: str) -> bool:
        """
        Ask LLM to finalize the step: ensure required artifacts are saved.
        This is triggered by user command /next.
        """
        print("\n🧩 Finalizing current step... (saving required artifacts)")

        # Set finalization flag to prevent trimming during finalization
        self._in_finalization = True
        
        try:
            # A strong instruction that forces tool calls
            finalize_instruction = self._finalize_instruction(step_id)
            self._append_user(finalize_instruction)

            try:
                self._tool_chat_round(step_id=step_id, force_depth=self.max_tool_depth)
            except Exception as e:
                logger.exception(f"Finalize step failed: {e}")
                if self.event_logger:
                    self.event_logger.log_event(
                        event_type="step_finalize_error",
                        metadata={
                            "step_id": step_id,
                            "error": str(e),
                        }
                    )
                return False
            
            # validate step requirements
            if not self._is_step_satisfied(step_id):
                print(f"\n⚠️ Step {step_id} validation failed: required artifacts not saved.")
                if self.event_logger:
                    self.event_logger.log_event(
                        event_type="step_validation_failed",
                        metadata={
                            "step_id": step_id,
                            "step_name": STEP_NAMES.get(step_id),
                            "artifacts_saved": [a.get("artifact_type") for a in self.state.artifacts_by_step.get(step_id, [])],
                            "required_artifacts": list(STEP_REQUIRED_ARTIFACTS.get(step_id, set())),
                        }
                    )
                return False

            if self.event_logger:
                self.event_logger.log_step_exit(
                    step_id=step_id,
                    substep="chat",
                    success=True,
                    metadata={
                        "step_name": STEP_NAMES.get(step_id),
                        "artifacts": self.state.artifacts_by_step.get(step_id, []),
                    },
                )

            print(f"✅ Step {step_id} finalized and artifacts saved.")
            self._save_state()
            return True
        
        finally:
            # Always reset finalization flag
            self._in_finalization = False

    def _final_answer(self) -> None:
        """
        Ask LLM for a concise final summary and call final_answer tool.
        """
        print("\n🧾 Writing final answer...")

        instruction = (
            "You have completed all steps. Provide a concise final summary including:\n"
            "- treatment, outcome\n"
            "- estimation method\n"
            "- ATE and CI (if available)\n"
            "- a short note on assumptions/limitations\n\n"
            "Then call final_answer(summary=..., all_steps_complete=true)."
        )
        self._append_user(instruction)
        self._tool_chat_round(step_id="3", force_depth=self.max_tool_depth)

        self.analysis_complete = True

    # -----------------------------
    # Tool-chat round (single or chained)
    # -----------------------------
    def _tool_chat_round(self, step_id: str, force_depth: Optional[int] = None) -> bool:
        """
        One assistant response + (optional) tool execution chain.
        Returns True if the chain completes without exception.
        """
        depth_budget = force_depth if force_depth is not None else self.max_tool_depth
        depth = 0

        while depth < depth_budget:
            msg = self._call_gpt(step_id=step_id)
            tool_calls = getattr(msg, "tool_calls", None)

            # assistant message (must be appended BEFORE tool results for correct ordering)
            self._append_assistant_from_message(msg)

            if not tool_calls:
                # If assistant produced plain text, print it
                if msg.content:
                    print(f"\n🤖 Assistant: {msg.content}")
                return True

            # execute tools - ensure all tool calls are executed and results appended
            executed_tool_call_ids = set()
            for tc in tool_calls:
                self._execute_tool_call(tc, step_id=step_id)
                executed_tool_call_ids.add(tc.id)
            
            # Verify all tool calls have responses before continuing
            # Find the assistant message we just added
            last_assistant_idx = None
            for i in range(len(self.messages) - 1, -1, -1):
                if self.messages[i].get("role") == "assistant" and self.messages[i].get("tool_calls"):
                    last_assistant_idx = i
                    break
            
            if last_assistant_idx is not None:
                assistant_msg = self.messages[last_assistant_idx]
                tool_calls_from_msg = assistant_msg.get("tool_calls", [])
                tool_call_ids_from_msg = {tc.get("id") for tc in tool_calls_from_msg if isinstance(tc, dict)}
                
                # Check if all tool calls have responses
                tool_responses = {}
                for i in range(last_assistant_idx + 1, len(self.messages)):
                    msg = self.messages[i]
                    if msg.get("role") == "tool":
                        tool_call_id = msg.get("tool_call_id")
                        if tool_call_id:
                            tool_responses[tool_call_id] = True
                
                missing_responses = tool_call_ids_from_msg - set(tool_responses.keys())
                if missing_responses:
                    logger.warning(
                        f"Some tool calls are missing responses: {missing_responses}. "
                        f"Executed: {executed_tool_call_ids}, Expected: {tool_call_ids_from_msg}"
                    )
            
            # After executing all tools, continue to next round to get assistant response
            # This ensures all tool results are followed by an assistant message
            depth += 1

        print("\n⚠️ Tool-call depth budget reached. If step finalize didn't complete, continue chatting and try /next again.")
        return True

    def _call_gpt(self, step_id: str):
        start_time = time.time()

        self._cleanup_tool_messages()

        if self.event_logger:
            self.event_logger.log_llm_call_start(
                model=self.model,
                agent_name="baseline_agent",
                operation="chat_with_tools",
                step_id=step_id,
                metadata={"messages": len(self.messages)},
            )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.3,
            )
            duration = time.time() - start_time
            msg = resp.choices[0].message

            if self.event_logger:
                self.event_logger.log_llm_call_end(
                    model=self.model,
                    agent_name="baseline_agent",
                    operation="chat_with_tools",
                    duration=duration,
                    token_count=resp.usage.total_tokens if resp.usage else None,
                    success=True,
                    step_id=step_id,
                    metadata={"has_tool_calls": msg.tool_calls is not None},
                )
            return msg

        except Exception as e:
            duration = time.time() - start_time
            logger.exception("GPT call failed: %s", e)
            if self.event_logger:
                self.event_logger.log_llm_call_end(
                    model=self.model,
                    agent_name="baseline_agent",
                    operation="chat_with_tools",
                    duration=duration,
                    success=False,
                    error=str(e),
                    step_id=step_id,
                )
            raise

    # -----------------------------
    # Tool execution + artifact/state tracking
    # -----------------------------
    def _execute_tool_call(self, tool_call, step_id: str) -> None:
        tool_name = tool_call.function.name
        
        try:
            tool_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.exception(f"Failed to parse tool arguments for {tool_name}")
            result = {"success": False, "error": f"Invalid JSON arguments: {str(e)}"}
            self._append_tool_result(tool_call.id, tool_name, result)
            print(f"\n🔧 Tool: {tool_name}")
            print(f"   ✗ {self._format_tool_result(tool_name, result)}")
            return

        print(f"\n🔧 Tool: {tool_name}")
        # print small preview
        try:
            preview = json.dumps(tool_args, ensure_ascii=False)[:240]
        except Exception:
            preview = str(tool_args)[:240]
        print(f"   args: {preview}{'...' if len(preview) >= 240 else ''}")

        # Execute tool with error handling
        tool_func = BASELINE_TOOLS.get(tool_name)
        if not tool_func:
            result = {"success": False, "error": f"Unknown tool: {tool_name}"}
            logger.error(f"Unknown tool requested: {tool_name}")
        else:
            try:
                result = tool_func(**tool_args)
            except Exception as e:
                logger.exception(f"Tool execution failed: {tool_name}")
                result = {"success": False, "error": f"Tool execution error: {str(e)}"}
                
                # Log tool failure event
                if self.event_logger:
                    self.event_logger.log_event(
                        event_type="tool_execution_error",
                        metadata={
                            "tool_name": tool_name,
                            "step_id": step_id,
                            "error": str(e),
                        }
                    )

        # Append tool message
        try:
            self._append_tool_result(tool_call.id, tool_name, result)
        except Exception as e:
            logger.exception(f"Failed to append tool result for {tool_name}")
            # Create emergency fallback result
            fallback_result = {"success": False, "error": f"Failed to process result: {str(e)}"}
            self._append_tool_result(tool_call.id, tool_name, fallback_result)

        # Track last SQL query
        if tool_name == "run_sql" and result.get("success"):
            # Extract SQL from tool context
            from baseline.tools import _tool_context
            if _tool_context.get("last_sql_result"):
                self.state.last_sql_query = _tool_context["last_sql_result"].get("sql")
                self._save_state()
        
        # handle tool outcomes (artifact tracking / completion)
        if tool_name == "save_artifact" and result.get("success"):
            try:
                self._on_artifact_saved(step_id=step_id, save_result=result)
            except Exception as e:
                logger.exception(f"Failed to track artifact: {e}")

        if tool_name == "final_answer":
            # baseline.tools.final_answer returns all_steps_complete
            self.analysis_complete = bool(result.get("all_steps_complete", False))

        # print short tool result
        status_icon = "✓" if result.get("success") else "✗"
        print(f"   {status_icon} {self._format_tool_result(tool_name, result)}")
    
    def _append_tool_result(self, tool_call_id: str, tool_name: str, result: Dict[str, Any]) -> None:
        """
        Helper to append tool result message.
        Validates that there's a preceding assistant message with matching tool_calls.
        Handles multiple tool_calls from the same assistant message.
        """
        # Validate message order: tool message must follow assistant message with tool_calls
        if not self.messages:
            logger.error("Cannot append tool result: no messages in history")
            return
        
        # Find the most recent assistant message with tool_calls
        assistant_msg = None
        for i in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                assistant_msg = msg
                break
        
        if not assistant_msg:
            logger.warning(
                f"Cannot append tool result: no assistant message with tool_calls found. "
                f"Appending anyway to prevent data loss."
            )
            # Still append to prevent data loss, but log warning
        
        # Check if this tool_call_id already has a response - remove it first
        for i in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[i]
            if msg.get("role") == "tool" and msg.get("tool_call_id") == tool_call_id:
                logger.warning(
                    f"Tool call ID {tool_call_id} already has a response. Removing old response..."
                )
                self.messages.pop(i)
                break
        
        # Verify tool_call_id exists in the assistant's tool_calls (if assistant_msg found)
        if assistant_msg:
            tool_calls = assistant_msg.get("tool_calls", [])
            tool_call_ids = {tc.get("id") for tc in tool_calls if isinstance(tc, dict)}
            
            if tool_call_id not in tool_call_ids:
                logger.warning(
                    f"Tool call ID {tool_call_id} not found in assistant's tool_calls. "
                    f"Available IDs: {tool_call_ids}. Appending anyway."
                )
        
        # Append tool result - always append to ensure all tool calls have responses
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": json.dumps(result, default=str, ensure_ascii=False),
            }
        )
    
    def _cleanup_tool_messages(self) -> None:
        """
        Remove tool messages that don't have preceding assistant with tool_calls.
        Only removes orphaned tool messages, not ones that are part of a valid sequence.
        """
        if not self.messages:
            return
        
        cleaned = []
        # Track all valid tool_call_ids from assistant messages
        valid_tool_call_ids = set()
        
        for i, msg in enumerate(self.messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Collect all tool_call_ids from this assistant message
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        valid_tool_call_ids.add(tc.get("id"))
                    elif hasattr(tc, "id"):
                        valid_tool_call_ids.add(tc.id)
                cleaned.append(msg)
            elif msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                # Only keep tool messages with valid tool_call_ids
                if tool_call_id in valid_tool_call_ids:
                    cleaned.append(msg)
                else:
                    logger.warning(f"Removing orphaned tool message with tool_call_id={tool_call_id} at index {i}")
            else:
                cleaned.append(msg)
        
        self.messages = cleaned

    def _on_artifact_saved(self, step_id: str, save_result: Dict[str, Any]) -> None:
        """
        Track artifact in state + set convenience paths.
        Also saves the last SQL query if available when saving dataset artifacts.
        """
        art = {
            "artifact_type": save_result.get("artifact_type"),
            "path": save_result.get("path"),
            "sha256": save_result.get("sha256"),
            "ts": time.time(),
        }
        
        # If saving a dataset artifact and we have a last SQL query, save it too
        atype = art["artifact_type"]
        if atype == "dataset" and self.state.last_sql_query:
            try:
                from baseline.tools import _tool_context
                artifact_manager = _tool_context.get("artifact_manager")
                if artifact_manager:
                    # Save SQL query as a separate artifact
                    sql_path = artifact_manager.save_artifact(
                        artifact_type="sql",
                        data=self.state.last_sql_query,
                        filename=f"step{step_id}_query.sql",
                        step_id=step_id,
                        metadata={"associated_dataset": save_result.get("path")}
                    )
                    # Add SQL artifact to the same step
                    sql_art = {
                        "artifact_type": "sql",
                        "path": sql_path,
                        "sha256": save_result.get("sha256"),  # Will be recalculated by artifact_manager
                        "ts": time.time(),
                    }
                    self.state.artifacts_by_step[step_id].append(sql_art)
                    
                    # Update convenience path if step 1
                    if step_id == "1" and not self.state.step1_sql_path:
                        self.state.step1_sql_path = sql_path
            except Exception as e:
                logger.warning(f"Failed to save associated SQL query: {e}")
        
        self.state.artifacts_by_step[step_id].append(art)

        path = art["path"]

        if step_id == "1":
            if atype == "sql":
                self.state.step1_sql_path = path
            if atype == "dataset":
                self.state.step1_dataset_path = path
        elif step_id == "2":
            if atype == "graph":
                self.state.step2_graph_path = path
            if atype == "graph_adj":
                self.state.step2_graph_adj_path = path
        elif step_id == "3":
            if atype == "ate":
                self.state.step3_ate_path = path

        self._save_state()

    def _is_step_satisfied(self, step_id: str) -> bool:
        """
        Step1: must include both sql and dataset
        Step2: must include either graph or graph_adj
        Step3: must include ate
        """
        saved_types = {a.get("artifact_type") for a in self.state.artifacts_by_step.get(step_id, [])}

        if step_id == "1":
            return "sql" in saved_types and "dataset" in saved_types
        if step_id == "2":
            return ("graph" in saved_types) or ("graph_adj" in saved_types)
        if step_id == "3":
            return "ate" in saved_types
        return False

    def _format_tool_result(self, tool_name: str, result: Dict[str, Any]) -> str:
        if not result.get("success"):
            return f"Failed: {result.get('error', 'Unknown error')}"
        if tool_name == "get_schema":
            return f"schema loaded (tables={result.get('table_count', 0)})"
        if tool_name == "run_sql":
            return f"sql ok (rows={result.get('row_count', 0)})"
        if tool_name == "run_python":
            outs = result.get("outputs", {})
            return f"python ok (outputs={list(outs.keys())[:6]})"
        if tool_name == "save_artifact":
            return f"saved {result.get('artifact_type')} -> {result.get('path')}"
        if tool_name == "final_answer":
            return f"final_answer: {result.get('message')}"
        return "ok"

    # -----------------------------
    # Message helpers
    # -----------------------------
    def _append_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

        # store first non-command user question (optional)
        if self.state.user_question is None:
            # avoid treating the forced finalize instruction as a "question"
            if not self._parse_command(text):
                self.state.user_question = text
                self._save_state()

    def _append_assistant_from_message(self, msg) -> None:
        """
        Preserve tool_calls ordering by attaching tool_calls into assistant message
        if present (OpenAI expects this structure in subsequent turns).
        """
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in tool_calls
                    ],
                }
            )
        else:
            self.messages.append({"role": "assistant", "content": msg.content or ""})

    def _trim_history(self) -> None:
        """
        Keep system messages + last N user/assistant/tool messages for step chat.
        This is a simple turn-based trimming (stable for experiments).
        Skip trimming during finalization to preserve context.
        """
        # Skip trimming during finalization
        if self._in_finalization:
            return
        
        # Keep all initial system messages
        sys_msgs = [m for m in self.messages if m["role"] == "system"]
        non_sys = [m for m in self.messages if m["role"] != "system"]

        if len(non_sys) <= self.history_keep_turns * 2:
            return

        # Keep the last K non-system messages
        kept = non_sys[-self.history_keep_turns * 2 :]
        self.messages = sys_msgs + kept
        
        # Clean up orphaned tool messages after trim
        self._cleanup_tool_messages()

    # -----------------------------
    # Command parsing / help
    # -----------------------------
    def _parse_command(self, user_input: str) -> Optional[str]:
        s = user_input.strip().lower()

        if s == CMD_HELP.lower():
            return "help"
        if s == CMD_NEXT.lower():
            return "next"
        if s == CMD_FINAL.lower():
            return "final"
        if s == CMD_QUIT.lower():
            return "quit"

        # aliases (already lowercase in check)
        if s in {alias.lower() for alias in ALIASES_NEXT}:
            return "next"
        if s in {alias.lower() for alias in ALIASES_FINAL}:
            return "final"
        if s in {alias.lower() for alias in ALIASES_QUIT}:
            return "quit"

        return None

    def _print_help(self) -> None:
        print("\n" + "-" * 70)
        print("Help")
        print("- /next : finalize current step (save required artifacts) and move to next step")
        print("- /final: (Step 3 only) write final summary and end session")
        print("- /quit : end session immediately")
        print("- You can also type: 'quit', 'bye', 'exit', '종료' to end session")
        print("-" * 70)

    # -----------------------------
    # Prompts for step headers / finalize instructions
    # -----------------------------
    def _step_header(self, step_id: str) -> str:
        if step_id == "1":
            return (
                "You are in STEP 1 (Data Exploration & Extraction).\n"
                "Goal: identify relevant tables and extract a clean analysis dataset.\n"
                "You may call tools freely (get_schema, run_sql, run_python).\n"
                "When the user types /next, you MUST save required artifacts:\n"
                "- save_artifact(type=sql, data_ref='last', step_id='1', filename='step1_final.sql')\n"
                "- save_artifact(type=dataset, data_ref='last', step_id='1', filename='step1_dataset.parquet')\n"
            )
        if step_id == "2":
            return (
                "You are in STEP 2 (Causal Graph Discovery).\n"
                "Goal: discover/specify a causal graph over variables in the dataset.\n"
                "Use run_python for causal discovery or construct a graph dict.\n"
                "When the user types /next, you MUST save ONE of:\n"
                "- save_artifact(type=graph, data_ref=<graph_json>, step_id='2', filename='graph_final.json')\n"
                "- OR save_artifact(type=graph_adj, data_ref=<adj_df_name>, step_id='2', filename='graph_final_adj.csv')\n"
            )
        if step_id == "3":
            return (
                "You are in STEP 3 (Causal Effect Estimation).\n"
                "Goal: estimate ATE using the graph (and appropriate adjustment).\n"
                "When the user types /next, you MUST save:\n"
                "- save_artifact(type=ate, data_ref=<ate_json>, step_id='3', filename='ate_result.json')\n"
                "After /next succeeds, the user will type /final to finish.\n"
            )
        return f"You are in STEP {step_id}."

    def _state_summary_for_step(self, step_id: str) -> str:
        """
        Minimal structured context passed at step entry.
        Keep this short—your actual continuity comes from artifacts/state.json, not chat history.
        """
        lines = []
        lines.append("Session State Summary (trusted):")
        if self.state.user_question:
            lines.append(f"- User question: {self.state.user_question}")
        lines.append(f"- DB: {self.state.db_id}")

        if self.state.step1_sql_path:
            lines.append(f"- Step1 SQL path: {self.state.step1_sql_path}")
        if self.state.step1_dataset_path:
            lines.append(f"- Step1 dataset path: {self.state.step1_dataset_path}")
        if self.state.step2_graph_path:
            lines.append(f"- Step2 graph path: {self.state.step2_graph_path}")
        if self.state.step2_graph_adj_path:
            lines.append(f"- Step2 graph_adj path: {self.state.step2_graph_adj_path}")
        if self.state.step3_ate_path:
            lines.append(f"- Step3 ATE path: {self.state.step3_ate_path}")

        # Step-specific reminder
        if step_id in {"2", "3"} and not self.state.step1_dataset_path:
            lines.append("- WARNING: Step1 dataset not found in state. You may need to redo Step1.")
        if step_id == "3" and not (self.state.step2_graph_path or self.state.step2_graph_adj_path):
            lines.append("- WARNING: Step2 graph not found in state. You may need to redo Step2.")

        return "\n".join(lines)

    def _finalize_instruction(self, step_id: str) -> str:
        if step_id == "1":
            return (
                "Finalize STEP 1 now.\n"
                "Requirements:\n"
                "1) Ensure you have executed the final SQL needed for analysis.\n"
                "2) Save BOTH artifacts exactly as:\n"
                "   - save_artifact(artifact_type='sql', data_ref='last', step_id='1', filename='step1_final.sql')\n"
                "   - save_artifact(artifact_type='dataset', data_ref='last', step_id='1', filename='step1_dataset.parquet')\n"
                "If something is missing, fix it using tools first. Do NOT move on until both saves succeed."
            )
        if step_id == "2":
            return (
                "Finalize STEP 2 now.\n"
                "Requirements:\n"
                "1) Produce a causal graph from the Step1 dataset.\n"
                "2) Save ONE artifact (choose one):\n"
                "   - save_artifact(artifact_type='graph', data_ref=<graph_dict_or_json>, step_id='2', filename='graph_final.json')\n"
                "   - OR save_artifact(artifact_type='graph_adj', data_ref=<adj_df_name>, step_id='2', filename='graph_final_adj.csv')\n"
                "If graph generation fails, adjust approach and try again. Do NOT move on until saving succeeds."
            )
        if step_id == "3":
            return (
                "Finalize STEP 3 now.\n"
                "Requirements:\n"
                "1) Estimate ATE using appropriate adjustment based on the discovered graph.\n"
                "2) Save:\n"
                "   - save_artifact(artifact_type='ate', data_ref=<ate_dict_or_json>, step_id='3', filename='ate_result.json')\n"
                "After saving, respond briefly that Step 3 is finalized and the user can type /final."
            )
        return f"Finalize STEP {step_id} now by saving required artifacts."

    # -----------------------------
    # Session finalize
    # -----------------------------
    def _finalize_session(self) -> None:
        if self.event_logger:
            self.event_logger.log_session_end(
                termination_reason="completed" if self.analysis_complete else "user_exit",
                metadata={
                    "completed_steps": [sid for sid in ["1", "2", "3"] if self._is_step_satisfied(sid)],
                    "state_path": str(self.state_path),
                    "analysis_complete": self.analysis_complete,
                },
            )

        # Ensure manifest is written
        if self.artifact_manager:
            try:
                self.artifact_manager.save_manifest()
            except Exception:
                pass

        print("\n✅ Session ended")
        print(f"- artifacts_dir: {self.run_context.artifacts_dir}")
        print(f"- state.json: {self.state_path}")
        if hasattr(self.run_context, "events_file"):
            print(f"- events: {self.run_context.events_file}")
        print("\n👋 Returning to terminal...")


def run_baseline_interactive(
    participant_id: str,
    session_id: str,
    db_id: str,
    run_context: Any,
    model: str = "gpt-4o-mini",
) -> None:
    agent = BaselineAgent(
        participant_id=participant_id,
        session_id=session_id,
        db_id=db_id,
        run_context=run_context,
        model=model,
    )
    agent.run_interactive()