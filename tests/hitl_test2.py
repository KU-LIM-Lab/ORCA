
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
import json


# 1. State Schema 정의 (TypedDict)
# 이것이 그래프 전체에서 사용할 state의 "타입 정의"
# total=False: 모든 필드가 선택적 (AgentState와 동일하게)
class SimpleState(TypedDict, total=False):
    query: str
    analysis_mode: Optional[str]
    ground_truth_path: Optional[str]
    input_needed: Optional[str]
    plan_created: bool
    plan: str
    execution_result: Optional[str]
    check_gt: Optional[bool]


# 2. Planner 노드 - state를 받아서 수정하고 반환
# 중요: 반환된 state는 LangGraph가 자동으로 기존 state와 merge함!
def planner_node(state: SimpleState) -> dict:
    """Simple planner that asks user for input via interrupt"""
    print("\n=== PLANNER NODE ===")
    print(f"Current query: {state['query']}")
    
    # If analysis_mode already set, skip interrupt
    if state.get("analysis_mode"):
        print(f"✅ Analysis mode already set: {state['analysis_mode']}")
        state["plan_created"] = True
        state["plan"] = ["step1: explore_data", "step2: analyze", "step3: report"]
        # 이 state를 반환하면 LangGraph가 자동으로 다음 노드에 전달!
        return state
    
    # First time - check ground truth
    else:
        if not state.get("ground_truth_path"):
            
            print("⚠️  No ground truth found. Interrupting to ask user...")
            payload = {
                "question": "Provide ground truth path or skip",
                "required_fields": ["ground_truth_path"],
                "hint": "Set ground_truth_path to a file path or leave empty to skip"
            }
            decision = interrupt(payload)
            print(f"📥 Received from user: {decision}")
            
            if decision:
                for k, v in decision.items():
                    state[k] = v
        else :
            state["check_gt"] = True
            if not state.get("input_needed"):
                payload ={"input_needed":"Provide input"}
                decision = interrupt(payload)
            else:
                state["plan"] = ["step1: explore_data", "step2: analyze", "step3: report"]
                state["plan_created"] = True
                state["planner_completed"] = True

    
    return state


# 3. Executor 노드 - planner가 반환한 state를 자동으로 받음!
def executor_node(state: SimpleState) -> dict:
    """Simple executor that runs the plan"""
    print("\n=== EXECUTOR NODE ===")
    print(f"Executing plan: ")
    
    result = "Executed 1 steps successfully"
    state["execution_result"] = result
    
    print(f"✅ {result}")
    return state


# 4. Routing function - state를 보고 다음 노드 결정
def should_continue(state: dict) -> str:
    """Decide if planner should continue or move to executor"""
    if state.get("plan_created"):
        return "executor"
    else:
        return "planner"


# 5. Graph 구축 - 여기서 State 관리 메커니즘이 설정됨!
def build_graph():
    """Build simple orchestration graph with HITL"""
    graph = StateGraph(SimpleState)
    
    # Add nodes
    # 각 노드 함수는 state를 받아서 state를 반환해야 함
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    
    # Set entry point
    graph.set_entry_point("planner")
    
    # Add conditional edge from planner
    # 조건부 엣지: should_continue 함수의 반환값에 따라 라우팅
    graph.add_conditional_edges(
        "planner",
        should_continue,  # state를 받아서 다음 노드 이름을 반환
        {
            "planner": "planner",  # Loop back if not ready
            "executor": "executor"  # Proceed if plan created
        }
    )
    
    # Executor goes to END
    graph.add_edge("executor", END)
    
    return graph



# 6. Main execution function
def run_simple_hitl_test():
    """Run the simple HITL test"""
    print("\n" + "="*60)
    print("🧪 SIMPLE HITL TEST WITH INTERRUPT")
    print("="*60)
    
    # Build and compile graph
    graph = build_graph()
    compiled = graph.compile(checkpointer=InMemorySaver())
    
    initial_state = {
        "query": "Analyze customer churn",
        "analysis_mode": None,
        "ground_truth_path": None,
        "input_needed": None,
        "plan_created": False,
        "planner_completed": False,
        "executer_completed": False,
        "plan": [],
        "execution_result": None,
        "check_gt": None
    }
    
    
    config = {"configurable": {"thread_id": "test_session_1"}}
    
    print("\n📍 Starting execution with initial state...")
    print(json.dumps(initial_state, indent=2))
    
    # Execute the graph with streaming
    print("\n" + "-"*60)
    print("🚀 GRAPH EXECUTION START")
    print("-"*60)
    print("💡 Note: This will handle MULTIPLE interrupts automatically!\n")

    # 첫 실행을 위한 initial input
    current_input = initial_state
    interrupt_count = 0
    
    # While loop로 감싸서 interrupt가 여러 번 발생해도 계속 처리
    while True:
        print(f"\n{'='*60}")
        print(f"🔄 Stream iteration (interrupt count: {interrupt_count})")
        print(f"{'='*60}")
        
        # Stream 실행 - interrupt 발생 시 자연스럽게 종료됨
        found_interrupt = False
        completed = False
        
        for step in compiled.stream(current_input, config=config):
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
                user_answer = input("> ")
                user_data = json.loads(user_answer)
                
                print(f"✅ Received: {json.dumps(user_data, indent=2)}")
                
                # State 업데이트 (invoke 대신 update_state 사용)
                compiled.update_state(config, user_data)
                
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
                    print(f"   - plan_created: {node_state.get('plan_created', 'N/A')}")
                    print(f"   - planner_completed: {node_state.get('planner_completed', 'N/A')}")
                    print(f"   - analysis_mode: {node_state.get('analysis_mode', 'N/A')}")
        
        # Stream이 정상 완료되었는지 확인 (interrupt 없이 끝났는지)
        if not found_interrupt:
            print("\n✅ Stream completed without interrupts!")
            completed = True
            break
        
        # Interrupt가 발생했으면 다음 iteration에서 재개
        # current_input을 None으로 설정하여 checkpoint에서 재개
        current_input = None
        print(f"\n🔄 Resuming from checkpoint after interrupt #{interrupt_count}...")
    
    print("\n" + "="*60)
    print(f"✅ GRAPH EXECUTION COMPLETED!")
    print(f"📊 Total interrupts handled: {interrupt_count}")
    print("="*60)

if __name__ == "__main__":
    # Test 1: Basic interrupt detection
    run_simple_hitl_test()