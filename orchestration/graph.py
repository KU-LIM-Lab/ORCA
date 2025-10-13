# orchestration/graph.py
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from core.state import AgentState, ExecutionStatus, create_initial_state
from orchestration.planner.agent import PlannerAgent
from orchestration.executor.agent import ExecutorAgent
from monitoring.metrics.collector import MetricsCollector
import json

class OrchestrationGraph:
    """Main orchestration graph that coordinates planner and executor"""
    
    def __init__(self, 
                 planner_config: Optional[Dict[str, Any]] = None,
                 executor_config: Optional[Dict[str, Any]] = None,
                 metrics_collector: Optional[MetricsCollector] = None,
                 orchestration_config: Optional[Dict[str, Any]] = None):
        self.metrics_collector = metrics_collector
        self.orchestration_config = orchestration_config or {}
        self.interactive = bool(self.orchestration_config.get("interactive", False))
        
        # Initialize agents
        self.planner = PlannerAgent(
            name="planner",
            config=planner_config,
            metrics_collector=metrics_collector
        )
        
        self.executor = ExecutorAgent(
            name="executor", 
            config=executor_config,
            metrics_collector=metrics_collector
        )
        
        # Build the graph
        self.graph = self._build_graph()
        self.compiled_graph = None 

    
    def _build_graph(self) -> StateGraph:
        """Build the orchestration graph with HITL support"""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("planner", self._planner_node)
        graph.add_node("executor", self._executor_node)
        
        # Set entry point
        graph.set_entry_point("planner")
        
        # Add edges
        graph.add_conditional_edges(
            "executor",
            self._route_after_execution,
            {
                # terminate graph on success or error
                "success": END,
                "error": END,
                "continue": "executor"
            }
        )

        graph.add_conditional_edges(
            "planner",  # planner 노드의 결과에 따라
            lambda x: x["plan_created"], # state의 'next_node' 값을 보고 판단
            {
                False: "planner",   # 'ask_user'이면 다시 planner로
                True: "executor"  # 'success'이면 executor로
            }
        )

        
        return graph
    
    def _planner_node(self, state: AgentState) -> AgentState:
        """Planner node execution"""
        result = self.planner.step(state)
        state.update(result)
        state["planner_completed"] = True
        return state

    
    def _executor_node(self, state: AgentState) -> AgentState:
        """Executor node execution"""
        result = self.executor.step(state)
        state.update(result)
        # state["executor_completed"] = True
        return state
    
    
    def _route_after_execution(self, state: AgentState) -> str:
        """Route after executor execution"""
        if state.get("error"):
            return "error"
        if state.get("executor_completed"):
            return "success"
        return "continue"
    
    def _generate_final_report(self, state: AgentState) -> Dict[str, Any]:
        """Generate final analysis report"""
        """ llm으로 결과 작성하는 specialist agent로 구성 예정 """
        execution_log = state.get("execution_log", [])
        results = state.get("results", {})
        
        report = {
            "summary": {
                "query": state.get("initial_query", ""),
                "status": "completed",
                "total_steps": len(execution_log),
                "successful_steps": len([log for log in execution_log if log.get("success", False)]),
                "execution_time": sum(log.get("duration", 0) for log in execution_log)
            },
            "results": results,
            "execution_log": execution_log
        }
        
        return report
    
    def compile(self):
        """Compile the graph with in-memory checkpointer"""
        self.compiled_graph = self.graph.compile(checkpointer=InMemorySaver()) #checkpointer=InMemorySaver()
        return self.compiled_graph
    
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None) -> AgentState:
        """Execute the orchestration with a user query.
        If interactive is False, run non-interactively and return final state via invoke.
        If interactive is True, stream with HITL prompts and return the last known state.
        Uses thread_id for resumable interrupts.
        """
        if not self.compiled_graph:
            self.compile()
        
        # Create initial state
        initial_state = create_initial_state(query)
        initial_state["interactive"] = self.interactive
        if context:
            initial_state.update(context)
        
        # Execute the graph
        config = {"configurable": {"thread_id": session_id or "default_session"}}
        
        if not self.interactive:
            result = self.compiled_graph.invoke(initial_state, config=config)
            return result

        print("\n--- Graph Execution Stream Starts ---")
        final_state = None
        current_input = initial_state
        interrupt_count = 0

        # 전체 대화/세션을 관리하는 최상위 루프
        # While loop로 감싸서 interrupt가 여러 번 발생해도 계속 처리
        while True:
            print(f"\n{'='*60}")
            print(f"🔄 Stream iteration (interrupt count: {interrupt_count})")
            print(f"{'='*60}")
            
            # Stream 실행 - interrupt 발생 시 자연스럽게 종료됨
            found_interrupt = False
            completed = False
            
            for step in self.compiled_graph.stream(current_input, config=config):
                step_name = list(step.keys())[0]
                
                # Interrupt 감지
                if step_name == '__interrupt__':
                    state_data = step[step_name]
                    interrupt_obj = state_data[0] if isinstance(state_data, tuple) else state_data
                    
                    interrupt_count += 1
                    print(f"\n⏸️  INTERRUPT #{interrupt_count} DETECTED!")
                    print(f"📋 Interrupt payload:")
                    print(json.dumps(interrupt_obj.value, indent=2))
                    print(f"💬 Please provide input (JSON format):")
                    
                    # 사용자 입력 받기
                    while True:
                        try:
                            user_answer = input("> ")
                            user_data = json.loads(user_answer)
                            user_data["hitl_executed"] = True 
                            break 
                        except json.JSONDecodeError:
                            print("잘못된 JSON 형식입니다. 다시 입력해주세요.")
                    
                    print(f"✅ Received: {json.dumps(user_data, indent=2)}")
                
                    # State 업데이트 (invoke 대신 update_state 사용)
                    self.compiled_graph.update_state(config, user_data)

                    updated_state = self.compiled_graph.update_state(config, user_data)
                    print(updated_state)
                    
                    found_interrupt = True
                    # stream이 자연스럽게 종료되므로 break 불필요
                    # 하지만 for loop을 빠져나가고 while loop에서 재시도
                    break
                else:
                    # 일반 노드 실행 완료
                    node_state = step[step_name]
                    print(f"\n✅ Node '{step_name}' completed")
                    
                    if isinstance(node_state, dict):
                        print(f"📊 Current state:")
                        print(f"   - planner_completed: {node_state.get(f'{step_name}_completed', 'N/A')}")
                        final_state = node_state
            
            # Stream이 정상 완료되었는지 확인 (interrupt 없이 끝났는지)
            if not found_interrupt:
                print("\n✅ Stream completed without interrupts!")
                completed = True
                break

            # Interrupt가 발생했으면 다음 iteration에서 재개
            # current_input을 None으로 설정하여 checkpoint에서 재개
            current_input = None
            print(f"\n🔄 Resuming from checkpoint after interrupt #{interrupt_count}...")
            
    
        print("\n--- Graph Execution Stream 종료 ---\n")
        return final_state
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestration status"""
        return {
            "planner_status": self.planner.get_capabilities(),
            "executor_status": self.executor.get_execution_status(),
            "graph_compiled": self.compiled_graph is not None
        }

def create_orchestration_graph(
    planner_config: Optional[Dict[str, Any]] = None,
    executor_config: Optional[Dict[str, Any]] = None,
    metrics_collector: Optional[MetricsCollector] = None,
    orchestration_config: Optional[Dict[str, Any]] = None
) -> OrchestrationGraph:
    """Create and return an orchestration graph"""
    return OrchestrationGraph(
        planner_config=planner_config,
        executor_config=executor_config,
        metrics_collector=metrics_collector,
        orchestration_config=orchestration_config,
    )
