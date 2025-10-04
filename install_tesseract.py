#!/usr/bin/env python3
"""
Script to help install Tesseract OCR for Windows.
"""

import subprocess
import sys
import platform
import os

def check_tesseract():
    """Check if Tesseract is already installed."""
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, check=True)
        print("✅ Tesseract is already installed:")
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_tesseract_windows():
    """Provide instructions for installing Tesseract on Windows."""
    print("🔧 Tesseract OCR Installation for Windows:")
    print("=" * 50)
    print()
    print("Option 1: Download from GitHub (Recommended)")
    print("1. Go to: https://github.com/UB-Mannheim/tesseract/wiki")
    print("2. Download the latest Windows installer")
    print("3. Run the installer and follow the setup wizard")
    print("4. Make sure to add Tesseract to your PATH during installation")
    print()
    print("Option 2: Using Chocolatey")
    print("1. Install Chocolatey if you haven't: https://chocolatey.org/install")
    print("2. Run: choco install tesseract")
    print()
    print("Option 3: Using Scoop")
    print("1. Install Scoop if you haven't: https://scoop.sh/")
    print("2. Run: scoop install tesseract")
    print()
    print("After installation, restart your terminal and run this script again to verify.")

def install_tesseract_linux():
    """Provide instructions for installing Tesseract on Linux."""
    print("🔧 Tesseract OCR Installation for Linux:")
    print("=" * 50)
    print()
    print("Ubuntu/Debian:")
    print("sudo apt-get update")
    print("sudo apt-get install tesseract-ocr")
    print()
    print("CentOS/RHEL/Fedora:")
    print("sudo yum install tesseract")
    print("# or for newer versions:")
    print("sudo dnf install tesseract")
    print()
    print("Arch Linux:")
    print("sudo pacman -S tesseract")

def install_tesseract_macos():
    """Provide instructions for installing Tesseract on macOS."""
    print("🔧 Tesseract OCR Installation for macOS:")
    print("=" * 50)
    print()
    print("Using Homebrew (Recommended):")
    print("brew install tesseract")
    print()
    print("Using MacPorts:")
    print("sudo port install tesseract4")

def main():
    """Main installation helper."""
    print("Tesseract OCR Installation Helper")
    print("=" * 40)
    print()
    
    # Check if already installed
    if check_tesseract():
        print("\n🎉 Tesseract is ready to use!")
        return
    
    print("❌ Tesseract OCR is not installed or not in PATH")
    print()
    
    # Provide platform-specific instructions
    system = platform.system().lower()
    
    if system == "windows":
        install_tesseract_windows()
    elif system == "linux":
        install_tesseract_linux()
    elif system == "darwin":  # macOS
        install_tesseract_macos()
    else:
        print(f"❓ Unsupported platform: {system}")
        print("Please visit: https://tesseract-ocr.github.io/tessdoc/Installation.html")
    
    print()
    print("📝 Note: OCR functionality is optional. Your RAG system will work without it,")
    print("   but OCR enables better text extraction from image-based PDFs.")

if __name__ == "__main__":
    main()
