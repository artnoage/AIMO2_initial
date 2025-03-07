import sys
import os

# Print system information
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

# Try to add the project root to the path
try:
    # Get the absolute path of the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"Added current directory to Python path: {current_dir}")
    
    # Try importing the modules
    print("\nTrying to import modules:")
    
    try:
        from grpo.config import RewardConfig
        print("✓ Successfully imported RewardConfig")
    except ImportError as e:
        print(f"✗ Failed to import RewardConfig: {e}")
    
    try:
        from grpo.dynamic_reward import DynamicReward
        print("✓ Successfully imported DynamicReward")
    except ImportError as e:
        print(f"✗ Failed to import DynamicReward: {e}")
    
    try:
        from utils.similarity_checker import SolutionSimilarityChecker
        print("✓ Successfully imported SolutionSimilarityChecker")
    except ImportError as e:
        print(f"✗ Failed to import SolutionSimilarityChecker: {e}")
        
    # Check if the grpo directory exists
    if os.path.exists(os.path.join(current_dir, 'grpo')):
        print("\ngrpo directory exists in current directory")
        print(f"Contents: {os.listdir(os.path.join(current_dir, 'grpo'))}")
    else:
        print("\ngrpo directory does not exist in current directory")
        
    # List all directories in the current path
    print("\nDirectories in current path:")
    for item in os.listdir(current_dir):
        if os.path.isdir(os.path.join(current_dir, item)):
            print(f"- {item}")
            
except Exception as e:
    print(f"Error during path setup: {e}")
