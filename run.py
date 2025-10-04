#!/usr/bin/env python3
"""
Production runner for FDS Docs System.
This is the main entry point for running the system in production.
"""

import sys
import os
import warnings
from pathlib import Path

# Disable ChromaDB telemetry BEFORE any imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"
os.environ["CHROMA_PERSIST_DIRECTORY"] = "./data/vectordb"

# Suppress all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    """Main entry point."""
    print("FDS Docs System - Production Mode")
    print("=" * 40)
    
    try:
        # Start the Enterprise RAG API server
        print("[OK] Starting Enterprise RAG API server...")
        
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        print("[OK] Importing modules...")
        import uvicorn
        
        print("[OK] Starting API server on 0.0.0.0:8080")
        print("[OK] API documentation: http://0.0.0.0:8080/docs")
        print("[OK] Press Ctrl+C to stop the server")
        print()
        
        # Start the server
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=8080,
            reload=False,
            log_level="info"
        )
        
    except Exception as e:
        print(f"[ERROR] Error starting API server: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()
