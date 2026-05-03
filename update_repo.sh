#!/bin/bash

# ============================================================================
# Face-to-BMI Repository Update Script
# Clones latest updates and overwrites app.py with new face crop display code
# ============================================================================

set -e  # Exit on error

echo "🚀 Starting Face-to-BMI Repository Update..."
echo ""

# Configuration
REPO_URL="https://github.com/BesteKarnibat/face-to-bmi.git"
REPO_DIR="face-to-bmi"
BACKUP_DIR="app_backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================================
# STEP 1: Check if repo already exists
# ============================================================================
echo "📁 Checking repository status..."

if [ -d "$REPO_DIR" ]; then
    echo "✓ Repository found at ./$REPO_DIR"
    echo "  Pulling latest updates..."
    cd "$REPO_DIR"
    git pull origin main --quiet || git pull origin master --quiet
    cd ..
else
    echo "📥 Cloning repository..."
    git clone "$REPO_URL" "$REPO_DIR" --quiet
    echo "✓ Repository cloned successfully"
fi

echo ""

# ============================================================================
# STEP 2: Create backup directory
# ============================================================================
echo "💾 Creating backup..."

if [ ! -d "$BACKUP_DIR" ]; then
    mkdir -p "$BACKUP_DIR"
fi

# Backup existing app.py if it exists
if [ -f "$REPO_DIR/app.py" ]; then
    cp "$REPO_DIR/app.py" "$BACKUP_DIR/app.py.backup_$TIMESTAMP"
    echo "✓ Existing app.py backed up to: $BACKUP_DIR/app.py.backup_$TIMESTAMP"
else
    echo "ℹ No existing app.py found (fresh installation)"
fi

echo ""

# ============================================================================
# STEP 3: Copy new app.py
# ============================================================================
echo "🔄 Updating app.py with new face crop display code..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -f "$SCRIPT_DIR/app.py" ]; then
    cp "$SCRIPT_DIR/app.py" "$REPO_DIR/app.py"
    echo "✓ app.py updated successfully"
else
    echo "❌ Error: app.py not found in script directory"
    echo "   Make sure app.py is in the same directory as this script"
    exit 1
fi

echo ""

# ============================================================================
# STEP 4: Copy requirements.txt
# ============================================================================
echo "📦 Updating requirements.txt..."

if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    cp "$SCRIPT_DIR/requirements.txt" "$REPO_DIR/requirements.txt"
    echo "✓ requirements.txt updated successfully"
else
    echo "⚠ Warning: requirements.txt not found, skipping..."
fi

echo ""

# ============================================================================
# STEP 5: Install dependencies
# ============================================================================
echo "📥 Installing dependencies..."

cd "$REPO_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

# Install requirements
if [ -f "requirements.txt" ]; then
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    echo "✓ Dependencies installed successfully"
else
    echo "⚠ Warning: requirements.txt not found, skipping dependency installation"
fi

cd ..

echo ""

# ============================================================================
# STEP 6: Summary & Next Steps
# ============================================================================
echo "═══════════════════════════════════════════════════════════════════"
echo "✅ Update Complete!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "📍 Repository location: $(pwd)/$REPO_DIR"
echo "📋 Updated files:"
echo "   • app.py (face crop display)"
echo "   • requirements.txt (dependencies)"
echo ""
echo "💾 Backup location: $(pwd)/$BACKUP_DIR/"
echo ""
echo "🚀 To run the app:"
echo ""
echo "   cd $REPO_DIR"
echo "   source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
echo "   streamlit run app.py"
echo ""
echo "📖 The app will open at: http://localhost:8501"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
