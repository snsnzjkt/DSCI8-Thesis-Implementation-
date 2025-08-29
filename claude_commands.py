# claude_commands.py - Execute commands and capture outputs for Claude
import subprocess
import sys
import json
from datetime import datetime

class ClaudeCommandRunner:
    """Run commands and format outputs for Claude"""
    
    def __init__(self):
        self.log_file = "claude_interactions.log"
    
    def run_command(self, command, description=""):
        """Run command and log results"""
        print(f"🚀 Running: {description or command}")
        
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            output = {
                "timestamp": datetime.now().isoformat(),
                "command": command,
                "description": description,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
            # Log the interaction
            self.log_interaction(output)
            
            # Print results
            if output["success"]:
                print("✅ Success!")
                if output["stdout"]:
                    print("📋 Output:", output["stdout"][:500], "..." if len(output["stdout"]) > 500 else "")
            else:
                print("❌ Failed!")
                if output["stderr"]:
                    print("🚨 Error:", output["stderr"])
            
            return output
            
        except subprocess.TimeoutExpired:
            print("⏰ Command timed out after 5 minutes")
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            print(f"💥 Exception: {e}")
            return {"success": False, "error": str(e)}
    
    def log_interaction(self, result):
        """Log interaction to file"""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(result, indent=2) + "\n" + "="*50 + "\n")

def main():
    runner = ClaudeCommandRunner()
    
    if len(sys.argv) > 1:
        command = " ".join(sys.argv[1:])
        runner.run_command(command)
    else:
        print("Usage: python claude_commands.py <command>")
        print("Example: python claude_commands.py 'python data/preprocess.py'")

if __name__ == "__main__":
    main()
