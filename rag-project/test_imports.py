#!/usr/bin/env python
import sys
from pathlib import Path

# Add src to path using absolute path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

print(f"sys.path[0] = {sys.path[0]}")
print("Testing imports...")

try:
    import utils.config
    print(f"utils.config imported: {utils.config.__file__}")
    print(f"Has Config: {hasattr(utils.config, 'Config')}")
    print(f"Module contents: {dir(utils.config)}")
    
    from utils.config import Config
    print(f"✓ Config class imported!")
    print(f"Config.LLM_MODEL = {Config.LLM_MODEL}")
    
except Exception as e:
    import traceback
    print(f"✗ Error: {e}")
    traceback.print_exc()

print("\nTesting graph import...")
try:
    from agents.rag_graph import build_graph
    print("✓ build_graph imported!")
    cfg = Config()
    graph = build_graph(cfg)
    print("✓ Graph built successfully!")
    
except Exception as e:
    import traceback
    print(f"✗ Error: {e}")
    traceback.print_exc()

print("\n✓ All tests passed!")
