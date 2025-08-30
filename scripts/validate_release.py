#!/usr/bin/env python3
"""
Release validation script for bayesdesign.

This script performs pre-release checks that mirror the GitHub Actions workflow
to help catch issues before creating a release.
"""

import os
import sys
import subprocess
import re
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=cwd
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_version_consistency():
    """Check that package version is consistent and ready for release."""
    print("🔍 Checking version consistency...")
    
    # Get package version
    sys.path.insert(0, 'src')
    try:
        from bed import __version__
        package_version = __version__
    except ImportError as e:
        print(f"❌ Failed to import package version: {e}")
        return False
    
    # Check if version has 'dev' suffix
    if 'dev' in package_version:
        print(f"❌ Package version contains 'dev': {package_version}")
        print("   Remove 'dev' from src/bed/__init__.py before release")
        return False
    
    print(f"✅ Package version: {package_version}")
    return True, package_version

def check_changelog(version):
    """Check that CHANGELOG.md has an entry for the current version."""
    print("🔍 Checking changelog entry...")
    
    try:
        with open('CHANGELOG.md', 'r') as f:
            changelog_content = f.read()
        
        # Look for version entry
        version_pattern = rf"## \[{re.escape(version)}\]"
        if not re.search(version_pattern, changelog_content):
            print(f"❌ No changelog entry found for version {version}")
            print(f"   Add '## [{version}] - YYYY-MM-DD' section to CHANGELOG.md")
            return False
        
        print(f"✅ Changelog entry found for version {version}")
        return True
    
    except FileNotFoundError:
        print("❌ CHANGELOG.md not found")
        return False

def run_tests():
    """Run the test suite."""
    print("🧪 Running test suite...")
    
    success, stdout, stderr = run_command("tox -e py")
    if not success:
        print("❌ Tests failed")
        print("STDERR:", stderr)
        return False
    
    print("✅ All tests passed")
    return True

def validate_notebooks():
    """Validate Jupyter notebooks in examples directory."""
    print("📓 Validating Jupyter notebooks...")
    
    examples_dir = Path("examples")
    if not examples_dir.exists():
        print("⚠️  No examples directory found, skipping notebook validation")
        return True
    
    notebooks = list(examples_dir.glob("*.ipynb"))
    if not notebooks:
        print("⚠️  No notebooks found in examples directory")
        return True
    
    for notebook in notebooks:
        if notebook.name == "dev.ipynb":
            print(f"⏭️  Skipping dev notebook: {notebook.name}")
            continue
        
        print(f"📓 Validating: {notebook.name}")
        cmd = f"jupyter nbconvert --to notebook --execute --inplace {notebook} --ExecutePreprocessor.timeout=300"
        success, stdout, stderr = run_command(cmd, cwd="examples")
        
        if not success:
            print(f"❌ Notebook validation failed: {notebook.name}")
            print("STDERR:", stderr)
            return False
        
        print(f"✅ Successfully executed: {notebook.name}")
    
    print("✅ All notebooks validated successfully")
    return True

def test_build():
    """Test package building."""
    print("🏗️  Testing package build...")
    
    # Clean previous builds
    run_command("rm -rf dist/ build/")
    
    success, stdout, stderr = run_command("python -m build")
    if not success:
        print("❌ Build failed")
        print("STDERR:", stderr)
        return False
    
    # Check build artifacts
    dist_path = Path("dist")
    if not dist_path.exists():
        print("❌ No dist directory created")
        return False
    
    artifacts = list(dist_path.glob("*"))
    if len(artifacts) < 2:  # Should have both .tar.gz and .whl
        print("❌ Expected both wheel and source distribution")
        return False
    
    print("✅ Package built successfully")
    print(f"   Artifacts: {[a.name for a in artifacts]}")
    return True

def main():
    """Run all validation checks."""
    print("🚀 bayesdesign Release Validation")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("src/bed/__init__.py").exists():
        print("❌ Not in bayesdesign repository root")
        sys.exit(1)
    
    checks = [
        ("Version Consistency", check_version_consistency),
        ("Test Suite", run_tests),
        ("Notebook Validation", validate_notebooks),
        ("Package Build", test_build),
    ]
    
    results = {}
    package_version = None
    
    for name, check_func in checks:
        try:
            if name == "Version Consistency":
                success, package_version = check_func()
                results[name] = success
            else:
                results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} check failed with exception: {e}")
            results[name] = False
    
    # Check changelog if we have a version
    if package_version and results["Version Consistency"]:
        results["Changelog Entry"] = check_changelog(package_version)
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Validation Summary")
    print("=" * 40)
    
    all_passed = True
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All checks passed! Ready to release v{package_version}")
        print(f"Next steps:")
        print(f"  1. git tag v{package_version}")
        print(f"  2. git push --tags")
        print(f"  3. Create GitHub release")
        sys.exit(0)
    else:
        print("\n❌ Some checks failed. Please fix issues before releasing.")
        sys.exit(1)

if __name__ == "__main__":
    main()