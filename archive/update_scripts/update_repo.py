#!/usr/bin/env python3
"""
Face-to-BMI Repository Update Script
Clones latest updates and overwrites app.py with new face crop display code
Cross-platform (Windows, macOS, Linux)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
REPO_URL = "https://github.com/BesteKarnibat/face-to-bmi.git"
REPO_DIR = "face-to-bmi"
BACKUP_DIR = "archive/app_backups"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(text):
    """Print a formatted header."""
    print(f"\n{'=' * 67}")
    print(f"  {text}")
    print(f"{'=' * 67}\n")

def print_step(number, text):
    """Print a step indicator."""
    print(f"📌 STEP {number}: {text}")
    print("-" * 67)

def print_success(text):
    """Print success message."""
    print(f"✅ {text}")

def print_info(text):
    """Print info message."""
    print(f"ℹ️  {text}")

def print_warning(text):
    """Print warning message."""
    print(f"⚠️  {text}")

def print_error(text):
    """Print error message."""
    print(f"❌ {text}")

def run_command(cmd, description="", quiet=False):
    """
    Run a shell command safely.
    
    Args:
        cmd: Command to run (list or string)
        description: Description of what's being done
        quiet: If True, suppress output
    
    Returns:
        bool: True if successful, False otherwise
    """
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    if description:
        print(f"   Running: {' '.join(cmd)}")
    
    try:
        if quiet:
            subprocess.run(cmd, check=True, capture_output=True)
        else:
            subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        if description:
            print_error(f"{description} failed")
        return False
    except FileNotFoundError:
        if description:
            print_error(f"Command not found: {cmd[0]}")
        return False

def get_script_dir():
    """Get the directory where this script is located."""
    return str(Path(__file__).resolve().parent)

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def check_git_installed():
    """Check if git is installed."""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def clone_or_update_repo():
    """Clone repository if it doesn't exist, otherwise update it."""
    print_step(1, "Repository Setup")
    
    if os.path.isdir(REPO_DIR):
        print_info(f"Repository found at ./{REPO_DIR}")
        print("   Pulling latest updates...")
        os.chdir(REPO_DIR)
        
        # Try main branch first, then master
        if not run_command("git pull origin main", quiet=True):
            print("   Trying master branch...")
            run_command("git pull origin master", quiet=True)
        
        os.chdir("..")
        print_success("Repository updated")
    else:
        print(f"📥 Cloning repository from {REPO_URL}...")
        if run_command(f"git clone {REPO_URL} {REPO_DIR}"):
            print_success(f"Repository cloned to ./{REPO_DIR}")
        else:
            print_error("Failed to clone repository")
            return False
    
    return True

def create_backup():
    """Create backup of existing app.py."""
    print_step(2, "Backup Creation")
    
    # Create backup directory
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    app_path = os.path.join(REPO_DIR, "app.py")
    if os.path.isfile(app_path):
        backup_path = os.path.join(BACKUP_DIR, f"app.py.backup_{TIMESTAMP}")
        shutil.copy2(app_path, backup_path)
        print_success(f"Existing app.py backed up to: {backup_path}")
    else:
        print_info("No existing app.py found (fresh installation)")
    
    return True

def update_app_py():
    """Copy new app.py to repository."""
    print_step(3, "Update app.py")
    
    script_dir = get_script_dir()
    source_app = os.path.join(script_dir, "app.py")
    target_app = os.path.join(REPO_DIR, "app.py")
    
    if not os.path.isfile(source_app):
        print_error(f"app.py not found in {script_dir}")
        print_info("Make sure app.py is in the same directory as this script")
        return False
    
    try:
        shutil.copy2(source_app, target_app)
        print_success(f"app.py updated successfully")
        print(f"   Target: {target_app}")
        return True
    except Exception as e:
        print_error(f"Failed to copy app.py: {str(e)}")
        return False

def update_requirements():
    """Copy new requirements.txt to repository."""
    print_step(4, "Update requirements.txt")
    
    script_dir = get_script_dir()
    source_req = os.path.join(script_dir, "requirements.txt")
    target_req = os.path.join(REPO_DIR, "requirements.txt")
    
    if not os.path.isfile(source_req):
        print_warning("requirements.txt not found, skipping...")
        return True
    
    try:
        shutil.copy2(source_req, target_req)
        print_success(f"requirements.txt updated successfully")
        return True
    except Exception as e:
        print_error(f"Failed to copy requirements.txt: {str(e)}")
        return False

def install_dependencies():
    """Install Python dependencies in virtual environment."""
    print_step(5, "Dependency Installation")
    
    repo_path = os.path.abspath(REPO_DIR)
    venv_dir = os.path.join(repo_path, "venv")
    requirements_file = os.path.join(repo_path, "requirements.txt")
    
    # Check if requirements.txt exists
    if not os.path.isfile(requirements_file):
        print_warning("requirements.txt not found, skipping dependency installation")
        return True
    
    # Determine pip executable based on OS
    if sys.platform == "win32":
        pip_cmd = os.path.join(venv_dir, "Scripts", "pip")
    else:
        pip_cmd = os.path.join(venv_dir, "bin", "pip")
    
    # Create virtual environment if it doesn't exist
    if not os.path.isdir(venv_dir):
        print("   Creating virtual environment...")
        if not run_command(f"{sys.executable} -m venv {venv_dir}", quiet=True):
            print_warning("Failed to create virtual environment, skipping dependency installation")
            return True
    
    print("   Installing dependencies...")
    print("   (This may take a minute...)")
    
    # Upgrade pip
    run_command(f"{pip_cmd} install --upgrade pip", quiet=True)
    
    # Install requirements
    if run_command(f"{pip_cmd} install -r {requirements_file}", quiet=True):
        print_success("Dependencies installed successfully")
        return True
    else:
        print_warning("Some dependencies may have failed to install")
        return True  # Don't fail the entire script

def print_summary():
    """Print summary of changes."""
    print_header("✨ UPDATE COMPLETE!")
    
    repo_abs = os.path.abspath(REPO_DIR)
    backup_abs = os.path.abspath(BACKUP_DIR)
    
    print("📍 Repository location:")
    print(f"   {repo_abs}\n")
    
    print("📋 Updated files:")
    print(f"   • app.py (face crop display)")
    print(f"   • requirements.txt (dependencies)\n")
    
    print("💾 Backup location:")
    print(f"   {backup_abs}\n")
    
    print("🚀 To run the app:\n")
    
    if sys.platform == "win32":
        print(f"   cd {REPO_DIR}")
        print(f"   venv\\Scripts\\activate")
        print(f"   streamlit run app.py\n")
    else:
        print(f"   cd {REPO_DIR}")
        print(f"   source venv/bin/activate")
        print(f"   streamlit run app.py\n")
    
    print("📖 The app will open at: http://localhost:8501\n")
    print("=" * 67)

def main():
    """Main execution function."""
    print_header("🚀 Face-to-BMI Repository Update Script")
    
    # Check git installation
    if not check_git_installed():
        print_error("Git is not installed or not in PATH")
        print_info("Please install Git from https://git-scm.com/")
        return False
    
    print_success("Git is installed\n")
    
    # Execute steps
    steps = [
        clone_or_update_repo,
        create_backup,
        update_app_py,
        update_requirements,
        install_dependencies,
    ]
    
    for step in steps:
        if not step():
            print_error(f"Failed at {step.__name__}")
            return False
        print()
    
    # Print summary
    print_summary()
    print_success("All steps completed successfully!\n")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        sys.exit(1)
