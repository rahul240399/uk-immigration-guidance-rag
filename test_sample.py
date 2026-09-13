#!/usr/bin/env python3
"""
Test API discovery with a small sample to verify functionality
"""

import json
from src.api_discovery import ImmigrationAPIDiscovery

def test_sample():
    print("Testing API Discovery with small sample...")
    
    discovery = ImmigrationAPIDiscovery()
    
    # Test immigration rules discovery (first few)
    print("\n1. Testing Immigration Rules discovery...")
    immigration_rules = discovery.discover_immigration_rules()
    print(f"Found {len(immigration_rules)} immigration rules")
    for rule in immigration_rules[:3]:
        print(f"  - {rule}")
    
    # Test a few endpoints for accessibility
    print(f"\n2. Testing accessibility validation...")
    sample_endpoints = immigration_rules[:3] if len(immigration_rules) >= 3 else immigration_rules
    metadata = discovery.validate_accessibility(sample_endpoints)
    
    print(f"Validated {len(metadata)} endpoints:")
    for meta in metadata:
        status = "✓" if meta["accessible"] else "✗"
        print(f"  {status} {meta['endpoint_path']} - {meta.get('title', 'No title')}")
    
    print(f"\nSample test completed successfully!")

if __name__ == "__main__":
    test_sample()
