"""Quick start script to launch the GUI."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llwh.gui.main import main

if __name__ == '__main__':
    print("=" * 60)
    print("LARGE LANGUAGE-WORLD HYBRID AI")
    print("Revolutionary AI System")
    print("=" * 60)
    print("\nLaunching GUI application...")
    print("Please wait while the interface loads...")
    print()
    
    main()
