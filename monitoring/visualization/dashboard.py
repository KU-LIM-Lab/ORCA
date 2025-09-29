# core/metrics_dashboard.py
import json
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import seaborn as sns
from ..metrics.collector import MetricsCollector, MetricType

class MetricsDashboard:
    """Dashboard for visualizing metrics and performance data"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.collector = metrics_collector
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_performance_overview(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create performance overview dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ORCA Performance Overview', fontsize=16, fontweight='bold')
        
        # 1. Execution time by agent
        self._plot_execution_times(axes[0, 0])
        
        # 2. Memory usage over time
        self._plot_memory_usage(axes[0, 1])
        
        # 3. Token usage by agent
        self._plot_token_usage(axes[1, 0])
        
        # 4. Error rates
        self._plot_error_rates(axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_execution_times(self, ax):
        """Plot execution times by agent"""
        metrics = self.collector.metrics
        execution_metrics = [m for m in metrics if m.metric_type == MetricType.EXECUTION_TIME]
        
        if not execution_metrics:
            ax.text(0.5, 0.5, 'No execution time data', ha='center', va='center')
            ax.set_title('Execution Times by Agent')
            return
        
        # Group by agent
        agent_times = {}
        for metric in execution_metrics:
            agent = metric.agent_name
            if agent not in agent_times:
                agent_times[agent] = []
            agent_times[agent].append(metric.value)
        
        # Create box plot
        agents = list(agent_times.keys())
        times = [agent_times[agent] for agent in agents]
        
        ax.boxplot(times, labels=agents)
        ax.set_title('Execution Times by Agent')
        ax.set_ylabel('Time (seconds)')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_memory_usage(self, ax):
        """Plot memory usage over time"""
        snapshots = self.collector.performance_snapshots
        
        if not snapshots:
            ax.text(0.5, 0.5, 'No memory data', ha='center', va='center')
            ax.set_title('Memory Usage Over Time')
            return
        
        timestamps = [s.timestamp for s in snapshots]
        memory_usage = [s.process_memory_mb for s in snapshots]
        
        ax.plot(timestamps, memory_usage, linewidth=2, marker='o', markersize=4)
        ax.set_title('Memory Usage Over Time')
        ax.set_ylabel('Memory (MB)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_token_usage(self, ax):
        """Plot token usage by agent"""
        metrics = self.collector.metrics
        token_metrics = [m for m in metrics if m.metric_type == MetricType.TOKEN_COUNT]
        
        if not token_metrics:
            ax.text(0.5, 0.5, 'No token data', ha='center', va='center')
            ax.set_title('Token Usage by Agent')
            return
        
        # Group by agent
        agent_tokens = {}
        for metric in token_metrics:
            agent = metric.agent_name
            if agent not in agent_tokens:
                agent_tokens[agent] = 0
            agent_tokens[agent] += metric.value
        
        agents = list(agent_tokens.keys())
        tokens = list(agent_tokens.values())
        
        bars = ax.bar(agents, tokens)
        ax.set_title('Token Usage by Agent')
        ax.set_ylabel('Total Tokens')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, token_count in zip(bars, tokens):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tokens)*0.01,
                   f'{int(token_count)}', ha='center', va='bottom')
    
    def _plot_error_rates(self, ax):
        """Plot error rates by agent"""
        metrics = self.collector.metrics
        error_metrics = [m for m in metrics if m.metric_type == MetricType.ERROR_COUNT]
        
        if not error_metrics:
            ax.text(0.5, 0.5, 'No error data', ha='center', va='center')
            ax.set_title('Error Rates by Agent')
            return
        
        # Group by agent
        agent_errors = {}
        for metric in error_metrics:
            agent = metric.agent_name
            if agent not in agent_errors:
                agent_errors[agent] = 0
            agent_errors[agent] += metric.value
        
        agents = list(agent_errors.keys())
        errors = list(agent_errors.values())
        
        bars = ax.bar(agents, errors, color='red', alpha=0.7)
        ax.set_title('Error Counts by Agent')
        ax.set_ylabel('Error Count')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, error_count in zip(bars, errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(errors)*0.01,
                   f'{int(error_count)}', ha='center', va='bottom')
    
    def create_agent_detailed_view(self, agent_name: str, save_path: Optional[str] = None) -> plt.Figure:
        """Create detailed view for a specific agent"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Agent Performance: {agent_name}', fontsize=16, fontweight='bold')
        
        # Get agent-specific metrics
        agent_metrics = [m for m in self.collector.metrics if m.agent_name == agent_name]
        
        if not agent_metrics:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return fig
        
        # 1. Execution time over time
        self._plot_agent_execution_times(agent_metrics, axes[0, 0])
        
        # 2. Memory usage over time
        self._plot_agent_memory_usage(agent_metrics, axes[0, 1])
        
        # 3. Token usage over time
        self._plot_agent_token_usage(agent_metrics, axes[1, 0])
        
        # 4. Metric distribution
        self._plot_agent_metric_distribution(agent_metrics, axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_agent_execution_times(self, agent_metrics, ax):
        """Plot execution times for an agent over time"""
        execution_metrics = [m for m in agent_metrics if m.metric_type == MetricType.EXECUTION_TIME]
        
        if not execution_metrics:
            ax.text(0.5, 0.5, 'No execution time data', ha='center', va='center')
            ax.set_title('Execution Times Over Time')
            return
        
        timestamps = [m.timestamp for m in execution_metrics]
        times = [m.value for m in execution_metrics]
        
        ax.plot(timestamps, times, marker='o', linewidth=2, markersize=6)
        ax.set_title('Execution Times Over Time')
        ax.set_ylabel('Time (seconds)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_agent_memory_usage(self, agent_metrics, ax):
        """Plot memory usage for an agent over time"""
        memory_metrics = [m for m in agent_metrics if m.metric_type == MetricType.MEMORY_USAGE]
        
        if not memory_metrics:
            ax.text(0.5, 0.5, 'No memory data', ha='center', va='center')
            ax.set_title('Memory Usage Over Time')
            return
        
        timestamps = [m.timestamp for m in memory_metrics]
        memory = [m.value for m in memory_metrics]
        
        ax.plot(timestamps, memory, marker='s', linewidth=2, markersize=6, color='green')
        ax.set_title('Memory Usage Over Time')
        ax.set_ylabel('Memory (MB)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_agent_token_usage(self, agent_metrics, ax):
        """Plot token usage for an agent over time"""
        token_metrics = [m for m in agent_metrics if m.metric_type == MetricType.TOKEN_COUNT]
        
        if not token_metrics:
            ax.text(0.5, 0.5, 'No token data', ha='center', va='center')
            ax.set_title('Token Usage Over Time')
            return
        
        timestamps = [m.timestamp for m in token_metrics]
        tokens = [m.value for m in token_metrics]
        
        ax.plot(timestamps, tokens, marker='^', linewidth=2, markersize=6, color='orange')
        ax.set_title('Token Usage Over Time')
        ax.set_ylabel('Tokens')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_agent_metric_distribution(self, agent_metrics, ax):
        """Plot distribution of all metrics for an agent"""
        metric_types = {}
        for metric in agent_metrics:
            metric_type = metric.metric_type.value
            if metric_type not in metric_types:
                metric_types[metric_type] = []
            metric_types[metric_type].append(metric.value)
        
        if not metric_types:
            ax.text(0.5, 0.5, 'No metric data', ha='center', va='center')
            ax.set_title('Metric Distribution')
            return
        
        # Create box plot for each metric type
        types = list(metric_types.keys())
        values = [metric_types[t] for t in types]
        
        ax.boxplot(values, labels=types)
        ax.set_title('Metric Distribution')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
    
    def export_metrics_report(self, file_path: str):
        """Export comprehensive metrics report"""
        report = {
            "session_info": {
                "session_id": self.collector.session_id,
                "total_metrics": len(self.collector.metrics),
                "monitoring_duration": self.collector.get_metrics_summary().get("session_duration", 0)
            },
            "summary": self.collector.get_metrics_summary(),
            "agent_metrics": {},
            "performance_snapshots": [s.to_dict() for s in self.collector.performance_snapshots]
        }
        
        # Add agent-specific metrics
        agents = set(m.agent_name for m in self.collector.metrics)
        for agent in agents:
            report["agent_metrics"][agent] = self.collector.get_agent_metrics(agent)
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Metrics report exported to {file_path}")

def create_dashboard(metrics_collector: MetricsCollector) -> MetricsDashboard:
    """Create a metrics dashboard"""
    return MetricsDashboard(metrics_collector)
