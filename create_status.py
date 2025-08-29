# create_status.py - Fixed for Windows encoding
import json
import os
import sys
from datetime import datetime
from pathlib import Path

def create_project_status():
    """Create a comprehensive project status for Claude"""
    
    project_status = {
        "timestamp": datetime.now().isoformat(),
        "project_name": "SCS-ID Intrusion Detection",
        "structure": get_directory_structure(),
        "files_content": get_important_files(),
        "recent_outputs": get_recent_outputs(),
        "environment": get_environment_info(),
        "git_status": get_git_info(),
        "todo": get_todo_items()
    }
    
    # Save to file - FIXED: Added UTF-8 encoding
    with open("claude_project_status.json", "w", encoding='utf-8') as f:
        json.dump(project_status, f, indent=2, ensure_ascii=False)
    
    # Create a readable summary - FIXED: Added UTF-8 encoding
    create_readable_summary(project_status)
    
    print("Project status created successfully!")
    print("Share 'claude_summary.md' with Claude for full context")
    
    return project_status

def get_directory_structure():
    """Get project directory tree"""
    structure = {}
    
    for root, dirs, files in os.walk("."):
        # Skip hidden and cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        rel_root = os.path.relpath(root, ".")
        if rel_root == ".":
            rel_root = "root"
        
        # Get file info
        py_files = [f for f in files if f.endswith('.py')]
        other_files = [f for f in files if not f.endswith('.py') and not f.startswith('.') and f != 'claude_project_status.json']
        
        if py_files or other_files or dirs:
            structure[rel_root] = {
                "python_files": py_files,
                "other_files": other_files,
                "directories": dirs
            }
    
    return structure

def get_important_files():
    """Get content of important files"""
    important_files = {}
    
    # Files to always include
    key_files = [
        "config.py", "requirements.txt", "README.md",
        "data/preprocess.py", "models/baseline_cnn.py", 
        "experiments/train_baseline.py", "models/scs_id.py"
    ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    important_files[file_path] = {
                        "content": content,
                        "size": len(content),
                        "lines": len(content.splitlines()),
                        "exists": True
                    }
            except Exception as e:
                important_files[file_path] = {"error": str(e), "exists": False}
        else:
            important_files[file_path] = {"exists": False, "error": "File not found"}
    
    return important_files

def get_recent_outputs():
    """Get recent execution outputs"""
    outputs = {}
    
    # Check for common output locations
    output_locations = ["results", "logs", "outputs", "data"]
    
    for location in output_locations:
        if os.path.exists(location):
            try:
                files = os.listdir(location)
                recent_files = []
                
                for file in files:
                    file_path = os.path.join(location, file)
                    if os.path.isfile(file_path):
                        mtime = os.path.getmtime(file_path)
                        recent_files.append({
                            "name": file,
                            "modified": datetime.fromtimestamp(mtime).isoformat(),
                            "size": os.path.getsize(file_path)
                        })
                
                outputs[location] = sorted(recent_files, key=lambda x: x["modified"], reverse=True)[:5]
            except Exception as e:
                outputs[location] = {"error": str(e)}
    
    return outputs

def get_environment_info():
    """Get Python environment info"""
    try:
        import subprocess
        
        # Get Python version
        python_version = sys.version.split()[0]  # Just version number
        
        # Get key packages only (to avoid long output)
        key_packages = ['torch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib']
        installed_packages = {}
        
        for package in key_packages:
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "show", package], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    # Extract version from pip show output
                    for line in result.stdout.split('\n'):
                        if line.startswith('Version:'):
                            installed_packages[package] = line.split(': ')[1]
                            break
                else:
                    installed_packages[package] = "Not installed"
            except:
                installed_packages[package] = "Unknown"
        
        return {
            "python_version": python_version,
            "key_packages": installed_packages,
            "working_directory": os.getcwd()
        }
    except Exception as e:
        return {"error": str(e)}

def get_git_info():
    """Get git repository info"""
    try:
        import subprocess
        
        # Get current branch
        branch_result = subprocess.run(["git", "branch", "--show-current"], 
                                     capture_output=True, text=True)
        current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "Not a git repo"
        
        # Get status
        status_result = subprocess.run(["git", "status", "--porcelain"], 
                                     capture_output=True, text=True)
        git_status = status_result.stdout if status_result.returncode == 0 else "No git status"
        
        return {
            "current_branch": current_branch,
            "status": git_status.strip(),
            "has_changes": bool(git_status.strip())
        }
    except Exception as e:
        return {"error": "Git not available"}

def get_todo_items():
    """Extract TODO items from code"""
    todos = []
    
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                    for i, line in enumerate(lines):
                        if 'TODO' in line or 'FIXME' in line or 'BUG' in line:
                            todos.append({
                                "file": file_path,
                                "line": i + 1,
                                "content": line.strip()
                            })
                except:
                    pass
    
    return todos

def create_readable_summary(status):
    """Create a readable markdown summary for Claude - FIXED encoding"""
    
    summary = f"""# SCS-ID Project Status Report

**Generated:** {status['timestamp']}
**Project:** {status['project_name']}

## Project Structure
"""
    
    for path, info in status['structure'].items():
        summary += f"\n### {path}/\n"
        if info['python_files']:
            summary += f"**Python files:** {', '.join(info['python_files'])}\n"
        if info['other_files']:
            summary += f"**Other files:** {', '.join(info['other_files'])}\n"
        if info['directories']:
            summary += f"**Directories:** {', '.join(info['directories'])}\n"
    
    summary += f"""
## Environment
- **Python:** {status['environment'].get('python_version', 'Unknown')}
- **Working Directory:** {status['environment'].get('working_directory', 'Unknown')}

### Key Packages Status
"""
    
    if 'key_packages' in status['environment']:
        for package, version in status['environment']['key_packages'].items():
            summary += f"- **{package}:** {version}\n"
    
    summary += "\n## Key Files Status\n"
    
    for file_path, file_info in status['files_content'].items():
        if file_info.get('exists'):
            summary += f"- EXISTS **{file_path}**: {file_info.get('lines', 0)} lines\n"
        else:
            summary += f"- MISSING **{file_path}**: {file_info.get('error', 'Not found')}\n"
    
    if status['todo']:
        summary += f"\n## TODO Items ({len(status['todo'])} found)\n"
        for todo in status['todo'][:10]:  # Show first 10
            summary += f"- **{todo['file']}:{todo['line']}** - {todo['content']}\n"
    
    summary += "\n## Recent Activity\n"
    
    for location, files in status['recent_outputs'].items():
        if files and not isinstance(files, dict):
            summary += f"**{location}/**: {len(files)} recent files\n"
    
    summary += f"""
## Git Status
- **Branch:** {status['git_status'].get('current_branch', 'Unknown')}
- **Has Changes:** {status['git_status'].get('has_changes', False)}

---
*This report was generated for Claude to understand the current project state.*
"""
    
    # FIXED: Added UTF-8 encoding and error handling
    try:
        with open("claude_summary.md", "w", encoding='utf-8') as f:
            f.write(summary)
        print("✓ Claude summary created: claude_summary.md")
    except Exception as e:
        print(f"Error creating summary: {e}")
        # Fallback: create without emojis
        summary_clean = summary.replace("✓", "SUCCESS")
        with open("claude_summary.md", "w", encoding='utf-8') as f:
            f.write(summary_clean)

if __name__ == "__main__":
    try:
        create_project_status()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying simplified version...")
        
        # Fallback simple version
        simple_status = {
            "timestamp": datetime.now().isoformat(),
            "working_directory": os.getcwd(),
            "python_files": []
        }
        
        # Get Python files
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith('.py'):
                    simple_status["python_files"].append(os.path.join(root, file))
        
        with open("claude_project_status.json", "w", encoding='utf-8') as f:
            json.dump(simple_status, f, indent=2)
        
        print("Simple project status created successfully!")
