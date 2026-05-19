"""
Chunk 15: Phase - Base
Base phase class for pipeline phases
"""

from typing import Dict, Any
from abc import ABC, abstractmethod


class BasePhase(ABC):
    """Abstract base class for pipeline phases"""
    
    def __init__(self, config: Dict):
        """
        Initialize phase
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = None
    
    @abstractmethod
    def execute(self, context: Dict) -> Dict:
        """
        Execute phase and return results dict
        
        Args:
            context: Pipeline context dictionary
            
        Returns:
            Updated context dictionary
        """
        pass
    
    def validate_context(self, context: Dict, required_keys: list) -> bool:
        """
        Validate context has required keys
        
        Args:
            context: Context dictionary
            required_keys: List of required key names
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        missing = [k for k in required_keys if k not in context]
        if missing:
            raise ValueError(f"Context missing required keys: {missing}")
        return True


def validate_base_phase_interface():
    """
    Ensure BasePhase enforces interface
    
    Raises:
        AssertionError: If BasePhase doesn't enforce abstract method
    """
    try:
        # Try to instantiate abstract class (should fail)
        base = BasePhase({})
        base.execute({})
        raise AssertionError("Should have raised TypeError for abstract class")
    except TypeError:
        pass  # Expected - cannot instantiate abstract class


if __name__ == "__main__":
    # Self-test
    print("Testing BasePhase...")
    
    validate_base_phase_interface()
    print("[PASS] BasePhase correctly enforces abstract interface")
    
    print("\n[PASS] BasePhase tests passed")