# examples/metrics_usage.py
"""
Example usage of the metrics system
"""
import time
import random
from monitoring.metrics.collector import MetricsCollector, set_metrics_collector, track_execution_time, track_memory_usage
from monitoring.llm.tracker import track_llm_call, track_llm_generation, create_llm_tracker, record_llm_tokens
from monitoring.visualization.dashboard import create_dashboard
from core.base import BaseAgent, AgentType, SpecialistAgent

def main():
    """Example of how to use the metrics system"""
    
    # 1. Initialize metrics collector
    collector = MetricsCollector("demo_session_001")
    set_metrics_collector(collector)
    
    # Start background monitoring
    collector.start_monitoring(interval=0.5)
    
    # 2. Create agents with metrics collection
    data_explorer = DataExplorerAgent("data_explorer", collector)
    causal_discovery = CausalDiscoveryAgent("causal_discovery", collector)
    
    # 3. Simulate agent execution with metrics
    simulate_agent_execution(data_explorer)
    simulate_agent_execution(causal_discovery)
    
    # 4. Simulate LLM calls with token tracking
    simulate_llm_calls()
    
    # 5. Stop monitoring
    collector.stop_monitoring()
    
    # 6. Generate reports and visualizations
    print("\n=== Metrics Summary ===")
    summary = collector.get_metrics_summary()
    print(f"Total metrics: {summary['total_metrics']}")
    print(f"Session duration: {summary['session_duration']:.2f}s")
    
    print("\n=== Agent Metrics ===")
    for agent_name in ["data_explorer", "causal_discovery"]:
        agent_metrics = collector.get_agent_metrics(agent_name)
        print(f"\n{agent_name}:")
        print(f"  Total metrics: {agent_metrics['total_metrics']}")
        if 'execution_time' in agent_metrics['by_type']:
            exec_time = agent_metrics['by_type']['execution_time']
            print(f"  Avg execution time: {exec_time['average']:.2f}s")
        if 'token_count' in agent_metrics['by_type']:
            tokens = agent_metrics['by_type']['token_count']
            print(f"  Total tokens: {tokens['total']:.0f}")
    
    # 7. Create visualizations
    dashboard = create_dashboard(collector)
    
    # Save overview dashboard
    dashboard.create_performance_overview("metrics_overview.png")
    print("\nOverview dashboard saved to metrics_overview.png")
    
    # Save agent-specific dashboards
    for agent_name in ["data_explorer", "causal_discovery"]:
        dashboard.create_agent_detailed_view(agent_name, f"metrics_{agent_name}.png")
        print(f"Agent dashboard for {agent_name} saved to metrics_{agent_name}.png")
    
    # Export comprehensive report
    dashboard.export_metrics_report("metrics_report.json")
    print("Comprehensive report saved to metrics_report.json")

class DataExplorerAgent(SpecialistAgent):
    """Example Data Explorer Agent with metrics"""
    
    def __init__(self, name: str, metrics_collector):
        super().__init__(name, AgentType.SPECIALIST, metrics_collector=metrics_collector)
        self.set_domain_expertise(["data_analysis", "sql", "pandas"])
    
    def step(self, state):
        """Simulate data exploration step"""
        with track_execution_time(self.name, {"step": "data_exploration"}):
            # Simulate data processing
            time.sleep(random.uniform(0.5, 2.0))
            
            # Simulate memory usage
            with track_memory_usage(self.name, {"operation": "data_processing"}):
                # Simulate memory-intensive operation
                data = [random.random() for _ in range(10000)]
                processed_data = [x * 2 for x in data]
            
            # Simulate SQL generation
            self._generate_sql_query()
            
            return {"data_explored": True, "tables_found": 5}
    
    @track_llm_call("data_explorer", "gpt-4")
    def _generate_sql_query(self):
        """Simulate LLM call for SQL generation"""
        time.sleep(random.uniform(0.3, 1.0))
        # Simulate token usage
        record_llm_tokens("data_explorer", random.randint(100, 500), "gpt-4", "prompt")
        record_llm_tokens("data_explorer", random.randint(50, 200), "gpt-4", "completion")

class CausalDiscoveryAgent(SpecialistAgent):
    """Example Causal Discovery Agent with metrics"""
    
    def __init__(self, name: str, metrics_collector):
        super().__init__(name, AgentType.SPECIALIST, metrics_collector=metrics_collector)
        self.set_domain_expertise(["causal_inference", "statistics", "graph_theory"])
    
    def step(self, state):
        """Simulate causal discovery step"""
        with track_execution_time(self.name, {"step": "causal_discovery"}):
            # Simulate data diagnosis
            self._perform_data_diagnosis()
            
            # Simulate algorithm selection
            self._select_algorithms()
            
            # Simulate graph discovery
            self._discover_causal_graph()
            
            return {"causal_graph": "discovered", "algorithms_used": 3}
    
    @track_llm_generation("causal_discovery", "gpt-4")
    def _perform_data_diagnosis(self):
        """Simulate LLM call for data diagnosis"""
        time.sleep(random.uniform(0.8, 2.0))
        record_llm_tokens("causal_discovery", random.randint(200, 800), "gpt-4", "prompt")
        record_llm_tokens("causal_discovery", random.randint(100, 400), "gpt-4", "completion")
    
    def _select_algorithms(self):
        """Simulate algorithm selection"""
        with track_execution_time(self.name, {"operation": "algorithm_selection"}):
            time.sleep(random.uniform(0.2, 0.8))
    
    def _discover_causal_graph(self):
        """Simulate causal graph discovery"""
        with track_execution_time(self.name, {"operation": "graph_discovery"}):
            time.sleep(random.uniform(1.0, 3.0))
            # Simulate memory usage for graph processing
            with track_memory_usage(self.name, {"operation": "graph_processing"}):
                graph_data = [random.random() for _ in range(50000)]

def simulate_agent_execution(agent):
    """Simulate agent execution with metrics"""
    print(f"Executing {agent.name}...")
    
    # Simulate multiple execution steps
    for i in range(3):
        state = {"step": i, "data": f"sample_data_{i}"}
        try:
            result = agent.execute(state)
            print(f"  Step {i}: {result.success}")
        except Exception as e:
            print(f"  Step {i}: Error - {e}")

def simulate_llm_calls():
    """Simulate LLM calls with token tracking"""
    print("\nSimulating LLM calls...")
    
    # Simulate various LLM operations
    with create_llm_tracker("orchestrator", "plan_generation", "gpt-4") as tracker:
        time.sleep(random.uniform(0.5, 1.5))
        tracker.record_tokens(random.randint(300, 600), "prompt")
        tracker.record_tokens(random.randint(150, 300), "completion")
    
    with create_llm_tracker("data_explorer", "sql_optimization", "gpt-4") as tracker:
        time.sleep(random.uniform(0.3, 1.0))
        tracker.record_tokens(random.randint(200, 400), "prompt")
        tracker.record_tokens(random.randint(100, 200), "completion")

if __name__ == "__main__":
    main()
