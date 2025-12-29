"""
Analysis script for experiment runs.

Parses events.jsonl and computes experiment metrics for user study analysis.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd


def load_events(events_file: Path) -> List[Dict[str, Any]]:
    """Load events from JSONL file."""
    events = []
    with open(events_file, "r") as f:
        for line in f:
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError:
                continue
    return events


def compute_session_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute session-level metrics."""
    session_start = None
    session_end = None
    
    for event in events:
        if event["event_type"] == "session_start":
            session_start = datetime.fromisoformat(event["timestamp"])
        elif event["event_type"] == "session_end":
            session_end = datetime.fromisoformat(event["timestamp"])
    
    duration = None
    if session_start and session_end:
        duration = (session_end - session_start).total_seconds()
    
    termination_reason = "unknown"
    for event in reversed(events):
        if event["event_type"] == "session_end":
            termination_reason = event.get("data", {}).get("termination_reason", "unknown")
            break
    
    return {
        "session_start": session_start.isoformat() if session_start else None,
        "session_end": session_end.isoformat() if session_end else None,
        "duration_seconds": duration,
        "termination_reason": termination_reason,
        "total_events": len(events),
    }


def compute_step_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute step-level metrics."""
    step_metrics = {}
    
    for step_id in ["1", "2", "3"]:
        step_enters = [e for e in events if e["event_type"] == "step_enter" and e.get("step_id") == step_id]
        step_exits = [e for e in events if e["event_type"] == "step_exit" and e.get("step_id") == step_id]
        
        total_duration = 0
        for exit_event in step_exits:
            exit_data = exit_event.get("data", {})
            if "duration" in exit_data:
                total_duration += exit_data["duration"]
        
        step_metrics[f"step{step_id}"] = {
            "enter_count": len(step_enters),
            "exit_count": len(step_exits),
            "total_duration": total_duration,
        }
    
    return step_metrics


def compute_hitl_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute HITL interaction metrics."""
    hitl_prompts = [e for e in events if e["event_type"] == "hitl_prompt_shown"]
    hitl_decisions = [e for e in events if e["event_type"] == "hitl_decision"]
    
    decisions_by_type = {}
    for event in hitl_decisions:
        decision = event.get("data", {}).get("decision", "unknown")
        decisions_by_type[decision] = decisions_by_type.get(decision, 0) + 1
    
    return {
        "hitl_prompt_count": len(hitl_prompts),
        "hitl_decision_count": len(hitl_decisions),
        "decisions_by_type": decisions_by_type,
    }


def compute_tool_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute tool call metrics."""
    tool_starts = [e for e in events if e["event_type"] == "tool_call_start"]
    tool_ends = [e for e in events if e["event_type"] == "tool_call_end"]
    
    tool_counts = {}
    tool_durations = {}
    tool_failures = {}
    
    for event in tool_ends:
        data = event.get("data", {})
        tool_name = data.get("tool_name", "unknown")
        duration = data.get("duration", 0)
        success = data.get("success", True)
        
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        
        if tool_name not in tool_durations:
            tool_durations[tool_name] = []
        tool_durations[tool_name].append(duration)
        
        if not success:
            tool_failures[tool_name] = tool_failures.get(tool_name, 0) + 1
    
    # Compute averages
    tool_avg_durations = {tool: sum(durs) / len(durs) for tool, durs in tool_durations.items()}
    
    return {
        "tool_call_count": len(tool_ends),
        "unique_tools": len(tool_counts),
        "tool_counts": tool_counts,
        "tool_avg_durations": tool_avg_durations,
        "tool_failures": tool_failures,
    }


def compute_llm_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute LLM call metrics."""
    llm_ends = [e for e in events if e["event_type"] == "llm_call_end"]
    
    total_calls = len(llm_ends)
    total_tokens = 0
    total_duration = 0
    failures = 0
    
    for event in llm_ends:
        data = event.get("data", {})
        if data.get("token_count"):
            total_tokens += data["token_count"]
        if data.get("duration"):
            total_duration += data["duration"]
        if not data.get("success", True):
            failures += 1
    
    return {
        "llm_call_count": total_calls,
        "total_tokens": total_tokens,
        "total_llm_duration": total_duration,
        "llm_failures": failures,
    }


def analyze_run(run_dir: Path) -> Dict[str, Any]:
    """Analyze a single run directory."""
    events_file = run_dir / "events.jsonl"
    
    if not events_file.exists():
        return {"error": "events.jsonl not found"}
    
    # Load events
    events = load_events(events_file)
    
    if not events:
        return {"error": "No events found"}
    
    # Get basic info from first event
    first_event = events[0]
    run_info = {
        "run_id": first_event.get("run_id"),
        "participant_id": first_event.get("participant_id"),
        "condition": first_event.get("condition"),
        "task_id": first_event.get("task_id"),
    }
    
    # Compute metrics
    session_metrics = compute_session_metrics(events)
    step_metrics = compute_step_metrics(events)
    hitl_metrics = compute_hitl_metrics(events)
    tool_metrics = compute_tool_metrics(events)
    llm_metrics = compute_llm_metrics(events)
    
    return {
        **run_info,
        **session_metrics,
        "steps": step_metrics,
        "hitl": hitl_metrics,
        "tools": tool_metrics,
        "llm": llm_metrics,
    }


def analyze_runs_batch(runs_dir: Path) -> List[Dict[str, Any]]:
    """Analyze all runs in a directory."""
    results = []
    
    # Find all run directories (contains events.jsonl)
    for events_file in runs_dir.rglob("events.jsonl"):
        run_dir = events_file.parent
        print(f"Analyzing: {run_dir}")
        
        result = analyze_run(run_dir)
        results.append(result)
    
    return results


def export_to_csv(results: List[Dict[str, Any]], output_file: Path) -> None:
    """Export results to CSV."""
    # Flatten nested dictionaries for CSV
    rows = []
    
    for result in results:
        if "error" in result:
            continue
        
        row = {
            "run_id": result.get("run_id"),
            "participant_id": result.get("participant_id"),
            "condition": result.get("condition"),
            "task_id": result.get("task_id"),
            "duration_seconds": result.get("duration_seconds"),
            "termination_reason": result.get("termination_reason"),
            "total_events": result.get("total_events"),
        }
        
        # Add step metrics
        steps = result.get("steps", {})
        for step_name, step_data in steps.items():
            row[f"{step_name}_duration"] = step_data.get("total_duration", 0)
            row[f"{step_name}_enter_count"] = step_data.get("enter_count", 0)
        
        # Add HITL metrics
        hitl = result.get("hitl", {})
        row["hitl_prompt_count"] = hitl.get("hitl_prompt_count", 0)
        row["hitl_decision_count"] = hitl.get("hitl_decision_count", 0)
        decisions = hitl.get("decisions_by_type", {})
        for decision_type, count in decisions.items():
            row[f"hitl_{decision_type}"] = count
        
        # Add tool metrics
        tools = result.get("tools", {})
        row["tool_call_count"] = tools.get("tool_call_count", 0)
        row["unique_tools"] = tools.get("unique_tools", 0)
        
        # Add LLM metrics
        llm = result.get("llm", {})
        row["llm_call_count"] = llm.get("llm_call_count", 0)
        row["total_tokens"] = llm.get("total_tokens", 0)
        row["total_llm_duration"] = llm.get("total_llm_duration", 0)
        
        rows.append(row)
    
    # Create DataFrame and export
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results exported to: {output_file}")
    print(f"   Total runs: {len(df)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment runs")
    parser.add_argument(
        "--runs-dir",
        required=True,
        type=Path,
        help="Directory containing run directories"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV file"
    )
    args = parser.parse_args()
    
    print(f"Analyzing runs in: {args.runs_dir}")
    results = analyze_runs_batch(args.runs_dir)
    
    print(f"\nFound {len(results)} runs")
    
    # Export to CSV
    export_to_csv(results, args.output)
    
    # Print summary
    print("\n=== Summary ===")
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed runs:")
        for r in failed:
            print(f"  - {r.get('run_id', 'unknown')}: {r.get('error')}")


if __name__ == "__main__":
    main()

