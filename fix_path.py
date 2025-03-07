import os
import sys

def add_project_root_to_path():
    """
    Add the project root directory to the Python path.
    This ensures that modules like 'grpo' and 'utils' can be imported.
    """
    # Get the absolute path of the current directory (assumed to be project root)
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Add to Python path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added project root to Python path: {project_root}")
    else:
        print(f"Project root already in Python path: {project_root}")
    
    # Check if the grpo module can be imported
    try:
        import grpo
        print("Successfully imported grpo module")
    except ImportError as e:
        print(f"Failed to import grpo module: {e}")
        
        # Check if the directory exists
        grpo_dir = os.path.join(project_root, 'grpo')
        if os.path.exists(grpo_dir):
            print(f"The grpo directory exists at: {grpo_dir}")
            print(f"Contents: {os.listdir(grpo_dir)}")
        else:
            print(f"The grpo directory does not exist at: {grpo_dir}")
    
    return project_root

if __name__ == "__main__":
    # Add project root to path
    project_root = add_project_root_to_path()
    
    # Print the current Python path
    print("\nCurrent Python path:")
    for i, path in enumerate(sys.path):
        print(f"{i}: {path}")
