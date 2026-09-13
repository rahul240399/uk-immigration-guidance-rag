"""
Clean API Discovery for UK Immigration Rules and Caseworker Guidance

Uses GOV.UK Content API to discover endpoints and validate accessibility.
Generates discovered_endpoints.json and accessibility_metadata.json files.
"""

import json
import time
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import requests


class ImmigrationAPIDiscovery:
    """
    Discovers UK Immigration endpoints using GOV.UK Content API.
    
    Features:
    - Immigration Rules discovery via manual structure
    - Caseworker Guidance discovery via collections
    - Accessibility validation and metadata logging
    - Rate limiting (2 requests/second)
    """
    
    def __init__(self):
        self.base_url = "https://www.gov.uk/api/content"
        self.rate_limit = 0.5  # 500ms between requests (2/sec)
        self.last_request = 0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()
    
    def _api_request(self, path: str) -> Dict:
        """Make rate-limited API request"""
        self._rate_limit()
        url = f"{self.base_url}{path}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"API request failed for {path}: {e}")
            raise
    
    def discover_immigration_rules(self) -> List[str]:
        """
        Discover Immigration Rules endpoints using manual structure.
        
        Returns:
            List of immigration rules endpoint paths
        """
        self.logger.info("Discovering Immigration Rules endpoints")
        
        data = self._api_request("/guidance/immigration-rules")
        endpoints = []
        
        # Extract from manual structure
        details = data.get("details", {})
        if "child_section_groups" in details:
            for group in details["child_section_groups"]:
                for section in group.get("child_sections", []):
                    if path := section.get("base_path"):
                        endpoints.append(path)
        
        self.logger.info(f"Found {len(endpoints)} Immigration Rules endpoints")
        return endpoints
    
    def discover_caseworker_guidance(self) -> List[str]:
        """
        Discover Caseworker Guidance endpoints using collections.
        
        Returns:
            List of caseworker guidance endpoint paths
        """
        self.logger.info("Discovering Caseworker Guidance endpoints")
        
        # Main collections page
        collections = self._api_request("/government/collections/visas-and-immigration-operational-guidance")
        endpoints = []
        
        # Process linked documents
        for doc in collections.get("links", {}).get("documents", []):
            doc_path = doc.get("base_path", "")
            doc_type = doc.get("document_type", "")
            
            if doc_type == "document_collection":
                # Explore sub-collection
                try:
                    sub_data = self._api_request(doc_path)
                    sub_endpoints = self._extract_guidance_documents(sub_data)
                    endpoints.extend(sub_endpoints)
                    self.logger.info(f"Processed {doc.get('title')}: {len(sub_endpoints)} documents")
                except Exception as e:
                    self.logger.warning(f"Failed to process {doc_path}: {e}")
            
            elif self._is_guidance_document(doc_path):
                # Direct guidance document
                endpoints.append(doc_path)
        
        self.logger.info(f"Found {len(endpoints)} Caseworker Guidance endpoints")
        return endpoints
    
    def _extract_guidance_documents(self, collection_data: Dict) -> List[str]:
        """Extract guidance documents from a collection"""
        documents = []
        
        for doc in collection_data.get("links", {}).get("documents", []):
            doc_path = doc.get("base_path", "")
            if self._is_guidance_document(doc_path) and doc.get("document_type") != "document_collection":
                documents.append(doc_path)
        
        return documents
    
    def _is_guidance_document(self, path: str) -> bool:
        """Check if path is a guidance document"""
        return (
            path.startswith("/government/publications/") or 
            path.startswith("/guidance/")
        ) and not any(exclude in path for exclude in [
            "/statistics/", "/news/", "/speeches/", "/consultations/"
        ])
    
    def validate_accessibility(self, endpoints: List[str]) -> List[Dict]:
        """
        Validate endpoint accessibility and extract metadata.
        
        Args:
            endpoints: List of endpoint paths to validate
            
        Returns:
            List of accessibility metadata for each endpoint
        """
        self.logger.info(f"Validating accessibility for {len(endpoints)} endpoints")
        
        results = []
        for i, endpoint in enumerate(endpoints, 1):
            metadata = {
                "endpoint_path": endpoint,
                "api_url": f"{self.base_url}{endpoint}",
                "accessible": False,
                "status_code": None,
                "error": None,
                "content_id": None,
                "title": None,
                "document_type": None,
                "validated_at": datetime.utcnow().isoformat()
            }
            
            try:
                data = self._api_request(endpoint)
                metadata.update({
                    "accessible": True,
                    "status_code": 200,
                    "content_id": data.get("content_id"),
                    "title": data.get("title"),
                    "document_type": data.get("document_type")
                })
            except Exception as e:
                metadata.update({
                    "error": str(e),
                    "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
                })
            
            results.append(metadata)
            
            # Progress logging every 10 endpoints
            if i % 10 == 0:
                accessible_count = sum(1 for r in results if r["accessible"])
                self.logger.info(f"Validated {i}/{len(endpoints)} endpoints ({accessible_count} accessible)")
        
        return results
    
    def run_discovery(self) -> Tuple[Dict, List[Dict]]:
        """
        Run complete discovery process.
        
        Returns:
            Tuple of (discovered_endpoints, accessibility_metadata)
        """
        self.logger.info("Starting UK Immigration API Discovery")
        start_time = time.time()
        
        # Discover endpoints
        immigration_rules = self.discover_immigration_rules()
        caseworker_guidance = self.discover_caseworker_guidance()
        all_endpoints = immigration_rules + caseworker_guidance
        
        # Create discovered endpoints structure
        discovered_endpoints = {
            "immigration_rules": immigration_rules,
            "caseworker_guidance": caseworker_guidance,
            "summary": {
                "immigration_rules_count": len(immigration_rules),
                "caseworker_guidance_count": len(caseworker_guidance),
                "total_count": len(all_endpoints),
                "discovered_at": datetime.utcnow().isoformat()
            }
        }
        
        # Validate accessibility
        accessibility_metadata = self.validate_accessibility(all_endpoints)
        
        duration = time.time() - start_time
        accessible_count = sum(1 for meta in accessibility_metadata if meta["accessible"])
        
        self.logger.info(f"Discovery completed in {duration:.1f}s")
        self.logger.info(f"Total: {len(all_endpoints)}, Accessible: {accessible_count}")
        
        return discovered_endpoints, accessibility_metadata
    
    def save_results(self, discovered_endpoints: Dict, accessibility_metadata: List[Dict]):
        """Save results to JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save discovered endpoints
        endpoints_file = "discovered_endpoints.json"
        with open(endpoints_file, 'w') as f:
            json.dump(discovered_endpoints, f, indent=2)
        self.logger.info(f"Saved endpoints to {endpoints_file}")
        
        # Save accessibility metadata
        metadata_file = f"accessibility_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                "metadata": accessibility_metadata,
                "summary": {
                    "total_endpoints": len(accessibility_metadata),
                    "accessible_endpoints": sum(1 for m in accessibility_metadata if m["accessible"]),
                    "validation_timestamp": datetime.utcnow().isoformat()
                }
            }, f, indent=2)
        self.logger.info(f"Saved accessibility metadata to {metadata_file}")


def main():
    """Main entry point for API discovery"""
    discovery = ImmigrationAPIDiscovery()
    discovered_endpoints, accessibility_metadata = discovery.run_discovery()
    discovery.save_results(discovered_endpoints, accessibility_metadata)


if __name__ == "__main__":
    main()
