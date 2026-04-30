"""
Chunk 17: Phase 3 - Temporal Weighting
Temporal feature extraction and weighting phase
"""

import numpy as np
from typing import Dict

from chunk_15_phase_base import BasePhase
from chunk_02_utils_logging import Logger
from chunk_12_evaluation_evaluator import Evaluator
from chunk_14_models_trainer import ModelTrainer
from chunk_13_state_manager import StateManager
from chunk_07_data_temporal import (
    apply_temporal_weighting_strategy, 
    extract_temporal_features,
    validate_temporal_features
)


class Phase3_TemporalWeighting(BasePhase):
    """Phase 3: Temporal weighting and feature extraction"""
    
    def __init__(self, config: Dict):
        """
        Initialize Phase 3
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.logger = Logger(config)
        self.evaluator = Evaluator(config)
        self.state_manager = StateManager()
    
    def execute(self, context: Dict) -> Dict:
        """
        Execute Phase 3: Apply temporal weighting
        
        Args:
            context: Pipeline context with data from Phase 1
            
        Returns:
            Updated context with temporal weights
        """
        self.logger.log("Starting Phase 3: Temporal Weighting", 'info')
        
        # Validate input from Phase 1
        self._validate_input(context)
        
        X = context['X']
        y = context['y']
        dates = context['dates']
        
        self.logger.log(f"Processing {len(X)} samples with temporal weighting", 'info')
        
        # Extract temporal features
        temporal_features = extract_temporal_features(dates)
        self.logger.log(f"Extracted {len(temporal_features)} temporal features", 'info')
        
        # Apply temporal weighting
        strategy_config = {
            'type': 'linear',
            'multiplier': self.config.get('TEMPORAL_MULTIPLIER', 9.0)
        }
        
        temporal_weights = apply_temporal_weighting_strategy(dates, strategy_config)
        
        self.logger.log(
            f"Temporal weights: min={temporal_weights.min():.3f}, "
            f"max={temporal_weights.max():.3f}, mean={temporal_weights.mean():.3f}",
            'info'
        )
        
        # Store temporal features with weights
        temporal_features['weights'] = temporal_weights
        
        # Validate output
        validate_temporal_features(dates, temporal_features)
        
        # Update context
        context.update({
            'temporal_weights': temporal_weights,
            'temporal_features': temporal_features,
            'phase3_complete': True
        })
        
        self.logger.log("Phase 3 completed successfully", 'info')
        return context
    
    def _validate_input(self, context: Dict):
        """
        Validate Phase 3 input requirements
        
        Args:
            context: Input context
            
        Raises:
            ValueError: If validation fails
        """
        if not context.get('phase1_complete'):
            raise ValueError("Phase 1 must complete before Phase 3")
        
        required = ['X', 'y', 'dates']
        for key in required:
            if key not in context:
                raise ValueError(f"Phase 3 missing required input: {key}")


def validate_phase3_input(context: Dict) -> bool:
    """
    Ensure Phase 3 has required inputs from Phase 1
    
    Args:
        context: Input context
        
    Returns:
        True if valid
    """
    assert context.get('phase1_complete') == True, "Phase 1 must be complete"
    required = ['X', 'y', 'dates']
    for key in required:
        assert key in context, f"Phase 3 missing required input: {key}"
    return True


def validate_phase3_output(context: Dict) -> bool:
    """
    Validate Phase 3 output
    
    Args:
        context: Output context from Phase 3
        
    Returns:
        True if valid
    """
    assert 'temporal_weights' in context, "Missing temporal_weights"
    assert 'temporal_features' in context, "Missing temporal_features"
    
    X = context['X']
    weights = context['temporal_weights']
    
    assert len(weights) == len(X), "Weights length mismatch"
    assert np.all(weights > 0), "All weights must be positive"
    assert np.all(np.isfinite(weights)), "Weights must be finite"
    
    assert 'phase3_complete' in context, "Missing phase3_complete flag"
    assert context['phase3_complete'] == True
    
    return True


if __name__ == "__main__":
    # Self-test
    print("Testing Phase3_TemporalWeighting...")
    
    # Create test config
    config = {
        'TEMPORAL_MULTIPLIER': 9.0,
        'LOG_VERBOSITY': 0
    }
    
    # Create mock Phase 1 output
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    context = {
        'X': np.random.randn(n_samples, n_features).astype(np.float32),
        'y': np.random.randint(0, 2, n_samples),
        'dates': np.random.randint(20220101, 20230101, n_samples),
        'phase1_complete': True
    }
    
    # Run Phase 3
    phase3 = Phase3_TemporalWeighting(config)
    result = phase3.execute(context.copy())
    
    print(f"[PASS] Phase 3 executed successfully")
    print(f"   Temporal weights shape: {result['temporal_weights'].shape}")
    print(f"   Temporal features keys: {list(result['temporal_features'].keys())}")
    
    # Validate
    validate_phase3_output(result)
    print("[PASS] Phase 3 output validation passed")
    
    print("\n[PASS] Phase3_TemporalWeighting tests passed")