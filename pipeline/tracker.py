import datetime
from typing import List, Dict, Any

class PipelineStep:
    """Represents a single operation in the data processing pipeline."""
    def __init__(self, operation_name: str, stats: Dict[str, Any], timestamp: str = None):
        self.operation_name = operation_name
        self.stats = stats
        self.timestamp = timestamp or datetime.datetime.now().isoformat()
        
    def to_dict(self):
        return {
            "operation": self.operation_name,
            "stats": self.stats,
            "timestamp": self.timestamp
        }

class PipelineTracker:
    """Tracks the history of operations applied to a dataset."""
    def __init__(self):
        self.history: List[PipelineStep] = []
        
    def add_step(self, operation_name: str, stats: Dict[str, Any]):
        step = PipelineStep(operation_name, stats)
        self.history.append(step)
        
    def get_history(self) -> List[Dict[str, Any]]:
        """Returns the chronological audit log of all transformations."""
        return [step.to_dict() for step in self.history]
        
    def clear(self):
        self.history.clear()
