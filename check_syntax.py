"""Syntax check script - verifies all Python files compile correctly."""

import py_compile
import os
import sys


def check_syntax(filepath):
    """Check syntax of a Python file."""
    try:
        py_compile.compile(filepath, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)


def main():
    """Check syntax of all Python files."""
    print("=" * 60)
    print("SYNTAX CHECK - Large Language-World Hybrid AI")
    print("=" * 60)
    
    # Get all Python files
    python_files = []
    for root, dirs, files in os.walk('llwh'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Also check top-level files
    for file in ['examples.py', 'run_gui.py', 'setup.py']:
        if os.path.exists(file):
            python_files.append(file)
    
    print(f"\nChecking {len(python_files)} Python files...\n")
    
    passed = 0
    failed = 0
    
    for filepath in sorted(python_files):
        success, error = check_syntax(filepath)
        if success:
            print(f"✓ {filepath}")
            passed += 1
        else:
            print(f"✗ {filepath}")
            print(f"  Error: {error}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All files passed syntax check!")
        sys.exit(0)


if __name__ == '__main__':
    main()
