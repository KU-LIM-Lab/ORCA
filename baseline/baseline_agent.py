"""
Baseline Agent for User Study

Single GPT agent with OpenAI function calling for causal analysis.
This agent works through a 3-step workflow: Data → Graph → ATE
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

import openai
from openai import OpenAI

from prompts.baseline_prompts import BASELINE_SYSTEM_PROMPT
from baseline.tools import (
    get_schema, run_sql, run_python, save_artifact, final_answer,
    set_tool_context, BASELINE_TOOLS
)

logger = logging.getLogger(__name__)

# Tool schemas for OpenAI function calling
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": "Retrieve database schema including tables, columns, types, and relationships. Use this at the start to understand the database structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_id": {
                        "type": "string",
                        "description": "Database identifier (e.g., 'reef_db')"
                    }
                },
                "required": ["db_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute SQL query and return results. The result DataFrame is automatically stored as 'df' for use in Python code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_id": {
                        "type": "string",
                        "description": "Database identifier"
                    },
                    "sql": {
                        "type": "string",
                        "description": "SQL query to execute"
                    }
                },
                "required": ["db_id", "sql"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code safely. You have access to pandas, numpy, scipy, statsmodels, sklearn for causal analysis. The last SQL result is available as 'df'. Any DataFrames you create are stored for future use.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "context_vars": {
                        "type": "object",
                        "description": "Optional additional variables (rarely needed)",
                        "additionalProperties": True
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_artifact",
            "description": "Save an artifact to track analysis progress. Each step requires specific artifacts: Step 1 needs 'sql' and 'dataset', Step 2 needs 'graph' or 'graph_adj', Step 3 needs 'ate'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artifact_type": {
                        "type": "string",
                        "enum": ["sql", "dataset", "graph", "graph_adj", "ate", "estimation_spec", "schema"],
                        "description": "Type of artifact to save"
                    },
                    "data_ref": {
                        "type": "string",
                        "description": "Data reference: 'last' for SQL/dataset, variable name for graphs/ATE, or JSON string"
                    },
                    "step_id": {
                        "type": "string",
                        "enum": ["1", "2", "3"],
                        "description": "Step number (1=Data, 2=Graph, 3=ATE)"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional custom filename (auto-generated if not provided)"
                    }
                },
                "required": ["artifact_type", "step_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Signal completion of the causal analysis. Only use after all 3 steps are complete and artifacts are saved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of analysis results including ATE estimate"
                    },
                    "all_steps_complete": {
                        "type": "boolean",
                        "description": "Whether all 3 steps have been completed with artifacts"
                    }
                },
                "required": ["summary", "all_steps_complete"]
            }
        }
    }
]


class BaselineAgent:
    """Single GPT agent for causal analysis using OpenAI function calling."""
    
    def __init__(
        self,
        participant_id: str,
        session_id: str,
        db_id: str,
        run_context: Any,
        model: str = "gpt-4o-mini",
        max_iterations: int = 50
    ):
        """
        Initialize baseline agent.
        
        Args:
            participant_id: Participant identifier
            session_id: Session identifier
            db_id: Database identifier
            run_context: RunContext instance for experiment tracking
            model: OpenAI model to use
            max_iterations: Maximum number of conversation turns
        """
        self.participant_id = participant_id
        self.session_id = session_id
        self.db_id = db_id
        self.run_context = run_context
        self.model = model
        self.max_iterations = max_iterations
        
        # Get event logger and artifact manager
        self.event_logger = run_context.get_event_logger()
        self.artifact_manager = run_context.get_artifact_manager()
        
        # Set tool context for logging
        set_tool_context(
            event_logger=self.event_logger,
            artifact_manager=self.artifact_manager
        )
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Conversation history
        self.messages = [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT}
        ]
        
        # Step tracking
        self.step_status = {
            "1": {"completed": False, "artifacts": [], "entered": False},
            "2": {"completed": False, "artifacts": [], "entered": False},
            "3": {"completed": False, "artifacts": [], "entered": False}
        }
        self.current_step = "1"
        
        # Iteration counter
        self.iteration_count = 0
        
        # Completion flag
        self.analysis_complete = False
        
        logger.info(f"BaselineAgent initialized: participant={participant_id}, session={session_id}, db={db_id}")
    
    def run_interactive(self):
        """Run interactive baseline session."""
        print("\n" + "="*60)
        print("🤖 Baseline Causal Analysis System")
        print("="*60)
        print(f"   Participant ID: {self.participant_id}")
        print(f"   Session ID: {self.session_id}")
        print(f"   Database: {self.db_id}")
        print("="*60)
        print("\n💡 Instructions:")
        print("   • Describe your causal analysis question")
        print("   • The agent will guide you through 3 steps:")
        print("     1) Data Exploration & Extraction")
        print("     2) Causal Graph Discovery")
        print("     3) Causal Effect Estimation")
        print("   • Type 'quit' to exit")
        print()
        
        # Log session start
        if self.event_logger:
            self.event_logger.log_session_start(metadata={
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "db_id": self.db_id,
                "model": self.model
            })
        
        # Enter Step 1
        self._log_step_enter("1")
        
        # Main conversation loop
        while not self.analysis_complete and self.iteration_count < self.max_iterations:
            try:
                print("\n" + "-"*60)
                
                # Get user input
                user_input = input("\n🧑 You: ").strip()
                
                # Handle quit
                if user_input.lower() in ["quit", "exit", "q"]:
                    if self._confirm_quit():
                        break
                    else:
                        continue
                
                # Handle empty input
                if not user_input:
                    continue
                
                # Add user message
                self.messages.append({"role": "user", "content": user_input})
                self.iteration_count += 1
                
                # Process conversation turn
                self._process_turn()
                
                # Check if analysis is complete
                if self.analysis_complete:
                    print("\n✅ Causal analysis complete!")
                    break
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Session interrupted (Ctrl+C)")
                break
            except Exception as e:
                logger.exception(f"Error in conversation loop: {e}")
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'quit' to exit.")
        
        # Check for max iterations
        if self.iteration_count >= self.max_iterations:
            print(f"\n⚠️  Maximum iterations ({self.max_iterations}) reached.")
        
        # Log session end
        self._finalize_session()
        
        print("\n✅ Session completed")
        print(f"📁 Artifacts saved to: {self.run_context.artifacts_dir}")
        print(f"📝 Event log: {self.run_context.events_file}")
    
    def _process_turn(self):
        """Process one conversation turn with GPT and tool calls."""
        # Call GPT with tools
        response = self._call_gpt_with_tools()
        
        # Process response
        if response.tool_calls:
            # Execute tool calls
            for tool_call in response.tool_calls:
                self._execute_tool_call(tool_call)
            
            # Add assistant message with tool calls
            self.messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response.tool_calls
                ]
            })
            
            # Recursively process next turn (GPT will respond to tool results)
            self._process_turn()
        else:
            # No tool calls - display assistant response
            if response.content:
                print(f"\n🤖 Assistant: {response.content}")
                self.messages.append({"role": "assistant", "content": response.content})
    
    def _call_gpt_with_tools(self):
        """Call GPT with function calling."""
        start_time = time.time()
        
        # Log LLM call start
        if self.event_logger:
            self.event_logger.log_llm_call_start(
                model=self.model,
                agent_name="baseline_agent",
                operation="chat_with_tools",
                step_id=self.current_step,
                metadata={"iteration": self.iteration_count}
            )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.3
            )
            
            duration = time.time() - start_time
            message = response.choices[0].message
            
            # Log LLM call end
            if self.event_logger:
                self.event_logger.log_llm_call_end(
                    model=self.model,
                    agent_name="baseline_agent",
                    operation="chat_with_tools",
                    duration=duration,
                    token_count=response.usage.total_tokens if response.usage else None,
                    success=True,
                    step_id=self.current_step,
                    metadata={
                        "iteration": self.iteration_count,
                        "has_tool_calls": message.tool_calls is not None
                    }
                )
            
            return message
            
        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"GPT call failed: {e}")
            
            if self.event_logger:
                self.event_logger.log_llm_call_end(
                    model=self.model,
                    agent_name="baseline_agent",
                    operation="chat_with_tools",
                    duration=duration,
                    success=False,
                    error=str(e),
                    step_id=self.current_step
                )
            raise
    
    def _execute_tool_call(self, tool_call):
        """Execute a single tool call and add result to messages."""
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        print(f"\n🔧 Calling tool: {tool_name}")
        print(f"   Arguments: {json.dumps(tool_args, indent=2)[:200]}...")
        
        try:
            # Get tool function
            tool_func = BASELINE_TOOLS.get(tool_name)
            if not tool_func:
                result = {"success": False, "error": f"Unknown tool: {tool_name}"}
            else:
                # Execute tool
                result = tool_func(**tool_args)
            
            # Handle save_artifact results - track artifacts
            if tool_name == "save_artifact" and result.get("success"):
                self._track_artifact(result)
            
            # Handle final_answer - mark completion
            if tool_name == "final_answer":
                self.analysis_complete = result.get("all_steps_complete", False)
            
            # Format result for display
            print(f"   ✓ Result: {self._format_tool_result(tool_name, result)}")
            
            # Add tool result to messages
            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(result, default=str)
            })
            
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool_name}")
            error_result = {"success": False, "error": str(e)}
            
            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(error_result)
            })
            
            print(f"   ✗ Error: {e}")
    
    def _format_tool_result(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Format tool result for display."""
        if not result.get("success"):
            return f"Failed - {result.get('error', 'Unknown error')}"
        
        if tool_name == "get_schema":
            return f"Retrieved {result.get('table_count', 0)} tables"
        elif tool_name == "run_sql":
            return f"Query returned {result.get('row_count', 0)} rows"
        elif tool_name == "run_python":
            outputs = result.get("outputs", {})
            return f"Execution complete, {len(outputs)} output variables"
        elif tool_name == "save_artifact":
            return f"Saved {result.get('artifact_type')} for step {result.get('step_id')}"
        elif tool_name == "final_answer":
            return f"Analysis {result.get('message', 'complete')}"
        else:
            return "Success"
    
    def _track_artifact(self, save_result: Dict[str, Any]):
        """Track saved artifacts and check step completion."""
        artifact_type = save_result.get("artifact_type")
        step_id = save_result.get("step_id")
        
        if step_id and step_id in self.step_status:
            self.step_status[step_id]["artifacts"].append(artifact_type)
            
            # Check if step is complete
            self._check_step_completion(step_id)
    
    def _check_step_completion(self, step_id: str):
        """Check if a step is complete and transition if needed."""
        if self.step_status[step_id]["completed"]:
            return  # Already completed
        
        artifacts = self.step_status[step_id]["artifacts"]
        
        # Step 1: needs sql AND dataset
        if step_id == "1":
            if "sql" in artifacts and "dataset" in artifacts:
                self._complete_step("1")
                self._transition_to_step("2")
        
        # Step 2: needs graph OR graph_adj
        elif step_id == "2":
            if "graph" in artifacts or "graph_adj" in artifacts:
                self._complete_step("2")
                self._transition_to_step("3")
        
        # Step 3: needs ate
        elif step_id == "3":
            if "ate" in artifacts:
                self._complete_step("3")
                print("\n🎉 All 3 steps completed! You can now call final_answer.")
    
    def _complete_step(self, step_id: str):
        """Mark a step as complete."""
        if not self.step_status[step_id]["completed"]:
            self.step_status[step_id]["completed"] = True
            self._log_step_exit(step_id, success=True)
            print(f"\n✅ Step {step_id} completed!")
    
    def _transition_to_step(self, new_step_id: str):
        """Transition to a new step."""
        if new_step_id != self.current_step and new_step_id in self.step_status:
            self.current_step = new_step_id
            self._log_step_enter(new_step_id)
            
            step_names = {"1": "Data Exploration", "2": "Causal Graph Discovery", "3": "Effect Estimation"}
            print(f"\n📊 Moving to Step {new_step_id}: {step_names.get(new_step_id)}")
    
    def _log_step_enter(self, step_id: str):
        """Log step entry."""
        if self.event_logger and not self.step_status[step_id]["entered"]:
            self.event_logger.log_step_enter(
                step_id=step_id,
                substep="main",
                metadata={"step_name": self._get_step_name(step_id)}
            )
            self.step_status[step_id]["entered"] = True
    
    def _log_step_exit(self, step_id: str, success: bool = True):
        """Log step exit."""
        if self.event_logger:
            self.event_logger.log_step_exit(
                step_id=step_id,
                substep="main",
                success=success,
                metadata={
                    "step_name": self._get_step_name(step_id),
                    "artifacts": self.step_status[step_id]["artifacts"]
                }
            )
    
    def _get_step_name(self, step_id: str) -> str:
        """Get human-readable step name."""
        names = {
            "1": "Data Exploration & Extraction",
            "2": "Causal Graph Discovery",
            "3": "Causal Effect Estimation"
        }
        return names.get(step_id, f"Step {step_id}")
    
    def _confirm_quit(self) -> bool:
        """Ask user to confirm quit."""
        print("\n" + "="*60)
        print("⚠️  WARNING: Exiting will end your session!")
        print("="*60)
        print("   - All conversation history will be lost")
        print("   - Saved artifacts will be preserved")
        print()
        confirm = input("❓ Are you sure you want to quit? (yes/no): ").strip().lower()
        if confirm in ["yes", "y"]:
            return True
        else:
            print("✓ Continuing session...")
            return False
    
    def _finalize_session(self):
        """Finalize session and log completion."""
        # Check which steps were completed
        completed_steps = [sid for sid, status in self.step_status.items() if status["completed"]]
        
        # Log incomplete steps
        for step_id, status in self.step_status.items():
            if status["entered"] and not status["completed"]:
                self._log_step_exit(step_id, success=False)
        
        # Save artifact manifest
        if self.artifact_manager:
            self.artifact_manager.save_manifest()
        
        # Log session end
        if self.event_logger:
            self.event_logger.log_session_end(
                termination_reason="completed" if self.analysis_complete else "user_exit",
                metadata={
                    "completed_steps": completed_steps,
                    "total_iterations": self.iteration_count,
                    "all_steps_complete": len(completed_steps) == 3
                }
            )


def run_baseline_interactive(
    participant_id: str,
    session_id: str,
    db_id: str,
    run_context: Any
) -> None:
    """
    Run baseline interactive mode for user study.
    
    Args:
        participant_id: Unique participant identifier
        session_id: Session identifier
        db_id: Database identifier
        run_context: RunContext instance for experiment tracking
    """
    try:
        agent = BaselineAgent(
            participant_id=participant_id,
            session_id=session_id,
            db_id=db_id,
            run_context=run_context
        )
        
        agent.run_interactive()
        
    except Exception as e:
        logger.exception(f"Baseline session failed: {e}")
        print(f"\n❌ Fatal error: {e}")
        raise
