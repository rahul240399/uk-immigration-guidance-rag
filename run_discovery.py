#!/usr/bin/env python3
"""
Run UK Immigration API Discovery

Discovers endpoints and validates accessibility for:
- UK Immigration Rules
- Caseworker Guidance

Outputs:
- discovered_endpoints.json: List of discovered endpoints
- accessibility_metadata_YYYYMMDD_HHMMSS.json: Accessibility validation results
"""

from src.api_discovery import ImmigrationAPIDiscovery

def main():
    print("=" * 60)
    print("UK Immigration API Discovery")
    print("=" * 60)
    
    discovery = ImmigrationAPIDiscovery()
    discovered_endpoints, accessibility_metadata = discovery.run_discovery()
    discovery.save_results(discovered_endpoints, accessibility_metadata)
    
    # Display summary
    print("\n" + "=" * 60)
    print("DISCOVERY COMPLETE")
    print("=" * 60)
    summary = discovered_endpoints["summary"]
    print(f"Immigration Rules: {summary['immigration_rules_count']} endpoints")
    print(f"Caseworker Guidance: {summary['caseworker_guidance_count']} endpoints")
    print(f"Total: {summary['total_count']} endpoints")
    
    accessible = sum(1 for m in accessibility_metadata if m["accessible"])
    print(f"Accessible: {accessible}/{summary['total_count']} ({accessible/summary['total_count']*100:.1f}%)")
    
    print("\nFiles generated:")
    print("✓ discovered_endpoints.json")
    print(f"✓ accessibility_metadata_{discovery.logger.handlers[0].stream.name if discovery.logger.handlers else 'TIMESTAMP'}.json")

if __name__ == "__main__":
    main()
