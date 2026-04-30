"""
Chunk 13: State Manager
Pipeline state management and feedback loops
"""

from typing import Dict, Any, Optional
import time


class StateManager:
    """Manages pipeline state, feedback loops, and backtracking"""
    
    def __init__(self):
        """Initialize state manager"""
        self.feedback_loops = {
            'phase1_to_phase3': {'active': False, 'data': None},
            'phase2_to_phase3': {'active': False, 'data': None},
            'phase3_to_phase4': {'active': False, 'data': None},
            'phase4_to_phase2': {'active': False, 'data': None}
        }
        self.results_history = []
        self.backtracking_state = {
            'precision_history': [],
            'backtrack_count': 0,
            'max_backtracks': 5
        }
        self._context = {}  # Shared context storage
    
    def update_feedback_loop(self, from_phase: str, to_phase: str, data: Any):
        """
        Update feedback loop data
        
        Args:
            from_phase: Source phase name
            to_phase: Target phase name
            data: Data to pass through feedback loop
        """
        key = f'{from_phase}_to_{to_phase}'
        if key in self.feedback_loops:
            self.feedback_loops[key]['active'] = True
            self.feedback_loops[key]['data'] = data
    
    def get_feedback_data(self, from_phase: str, to_phase: str) -> Optional[Any]:
        """
        Retrieve feedback data
        
        Args:
            from_phase: Source phase name
            to_phase: Target phase name
            
        Returns:
            Feedback data if available, None otherwise
        """
        key = f'{from_phase}_to_{to_phase}'
        if key in self.feedback_loops and self.feedback_loops[key]['active']:
            return self.feedback_loops[key]['data']
        return None
    
    def is_feedback_active(self, from_phase: str, to_phase: str) -> bool:
        """
        Check if feedback loop is active
        
        Args:
            from_phase: Source phase name
            to_phase: Target phase name
            
        Returns:
            True if feedback loop is active
        """
        key = f'{from_phase}_to_{to_phase}'
        return self.feedback_loops.get(key, {}).get('active', False)
    
    def handle_backtracking(self, current_precision: float, 
                           previous_precision: Optional[float]) -> bool:
        """
        Check if backtracking is needed based on precision degradation
        
        Args:
            current_precision: Current precision score
            previous_precision: Previous precision score
            
        Returns:
            True if backtracking should occur
        """
        if previous_precision is None:
            return False
        
        # Backtrack if precision dropped by more than 5%
        if current_precision < previous_precision * 0.95:
            self.backtracking_state['backtrack_count'] += 1
            return True
        
        return False
    
    def store_results(self, phase: str, results: Dict):
        """
        Store phase results for history
        
        Args:
            phase: Phase name
            results: Results dictionary
        """
        self.results_history.append({
            'phase': phase,
            'results': results,
            'timestamp': time.time()
        })
        
        # Track precision for backtracking
        if 'precision' in results:
            self.backtracking_state['precision_history'].append(results['precision'])
    
    def get_context_value(self, key: str) -> Any:
        """
        Retrieve shared context data
        
        Args:
            key: Context key
            
        Returns:
            Context value
        """
        return self._context.get(key)
    
    def set_context_value(self, key: str, value: Any):
        """
        Update shared context
        
        Args:
            key: Context key
            value: Context value
        """
        self._context[key] = value
    
    def clear_context(self):
        """Clear all context data"""
        self._context.clear()
    
    def get_precision_history(self) -> list:
        """
        Get precision history for backtracking analysis
        
        Returns:
            List of precision scores
        """
        return self.backtracking_state['precision_history'].copy()
    
    def get_backtrack_count(self) -> int:
        """
        Get number of backtracks performed
        
        Returns:
            Backtrack count
        """
        return self.backtracking_state['backtrack_count']
    
    def should_stop_backtracking(self) -> bool:
        """
        Check if max backtracks reached
        
        Returns:
            True if should stop backtracking
        """
        return self.backtracking_state['backtrack_count'] >= self.backtracking_state['max_backtracks']


def validate_state_manager(state_manager: StateManager) -> bool:
    """
    Ensure StateManager functions correctly
    
    Args:
        state_manager: StateManager instance to validate
        
    Returns:
        True if valid
    """
    # Test feedback loop
    state_manager.update_feedback_loop('phase1', 'phase3', {'test': 'data'})
    assert state_manager.is_feedback_active('phase1', 'phase3'), "Feedback loop not activated"
    assert state_manager.get_feedback_data('phase1', 'phase3') == {'test': 'data'}, "Feedback data mismatch"
    
    # Test backtracking
    should_backtrack = state_manager.handle_backtracking(0.8, 0.9)
    assert isinstance(should_backtrack, bool), "Backtracking should return bool"
    assert should_backtrack == True, "Should backtrack when precision drops"
    
    # Test non-backtrack
    should_not_backtrack = state_manager.handle_backtracking(0.95, 0.9)
    assert should_not_backtrack == False, "Should not backtrack when precision improves"
    
    # Test context
    state_manager.set_context_value('test_key', 'test_value')
    assert state_manager.get_context_value('test_key') == 'test_value', "Context not stored"
    
    # Test results storage
    state_manager.store_results('test_phase', {'precision': 0.9})
    assert len(state_manager.results_history) > 0, "Results not stored"
    
    return True


if __name__ == "__main__":
    # Self-test
    print("Testing StateManager...")
    
    state_manager = StateManager()
    validate_state_manager(state_manager)
    print("[PASS] StateManager validated")
    
    print(f"   Backtrack count: {state_manager.get_backtrack_count()}")
    print(f"   Precision history: {state_manager.get_precision_history()}")
    print(f"   Results history length: {len(state_manager.results_history)}")
    
    print("\n[PASS] All StateManager tests passed")