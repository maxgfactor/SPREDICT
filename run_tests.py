#!/usr/bin/env python3
"""
CI/CD Test Runner
Runs all chunks in dependency order and validates the pipeline
"""

import subprocess
import sys
import os

# Change to the cicd directory
os.chdir('/home/laptop/projectai/spredict/workingfolder/cicd')

# Define test phases with their chunks
TEST_PHASES = {
    "Phase 0: Validation Framework": [
        "chunk_00_validation_framework.py"
    ],
    "Phase 1: Foundation Layer": [
        "chunk_01_config.py",
        "chunk_02_utils_logging.py",
        "chunk_03_utils_memory.py",
        "chunk_04_utils_metrics.py"
    ],
    "Phase 2: Data Layer": [
        "chunk_05_data_manager.py",
        "chunk_06_data_augmentation.py",
        "chunk_07_data_temporal.py"
    ],
    "Phase 3: Model Architectures": [
        "chunk_08_models_base.py",
        "chunk_09_models_advanced.py",
        "chunk_10_models_ensemble.py",
        "chunk_11_models_sklearn.py"
    ],
    "Phase 4: Training & Evaluation": [
        "chunk_12_evaluation_evaluator.py",
        "chunk_13_state_manager.py",
        "chunk_14_models_trainer.py"
    ],
    "Phase 5: Pipeline Phases": [
        "chunk_15_phase_base.py",
        "chunk_16_phase_1_setup.py",
        "chunk_17_phase_3_temporal.py",
        "chunk_18_phase_4_ensemble.py",
        "chunk_19_phase_5_optimization.py"
    ],
    "Phase 6: Orchestration": [
        "chunk_20_pipeline_main.py"
    ]
}


def run_test(file_path: str) -> bool:
    """Run a single test file"""
    try:
        result = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"  ✅ {file_path}")
            return True
        else:
            print(f"  ❌ {file_path}")
            print(f"     Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ⏱️  {file_path} (timeout)")
        return False
    except Exception as e:
        print(f"  ❌ {file_path} (exception: {e})")
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("🧪 CI/CD Pipeline Test Runner")
    print("="*70)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for phase_name, files in TEST_PHASES.items():
        print(f"\n{phase_name}")
        print("-"*70)
        
        for file_path in files:
            if os.path.exists(file_path):
                total_tests += 1
                if run_test(file_path):
                    passed_tests += 1
                else:
                    failed_tests.append(file_path)
                    print("\n⏹️  Stopping at first failure")
                    break
            else:
                print(f"  ⚠️  {file_path} (not found)")
                failed_tests.append(file_path)
                print("\n⏹️  Stopping at first failure")
                break
    
    # Summary
    print("\n" + "="*70)
    print("📊 Test Summary")
    print("="*70)
    print(f"Total:  {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\n❌ Failed tests:")
        for test in failed_tests:
            print(f"   - {test}")
        return 1
    else:
        print(f"\n✅ All tests passed!")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
