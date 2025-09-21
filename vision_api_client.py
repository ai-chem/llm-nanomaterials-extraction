import requests
import json
import time
from pathlib import Path
from typing import Dict, Optional


class VisionAPIClient:
    """Client for interacting with the Vision API"""
    
    def __init__(self, base_url: str = "http://77.234.216.102:17628"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def extract_file(
        self,
        file_path: str,
        dataset_type: str = "cytox",
        use_vlm: bool = True,
        max_pages: int = 0
    ) -> Dict:
        """
        Extract data from a PDF file using the Vision API.

        Args:
            file_path: Path to the PDF file.
            dataset_type: Extraction type (e.g., "cytox", "nanozymes", "magnetic").
            use_vlm: Whether to use the vision-language model.
            max_pages: Maximum number of pages to process (0 = all).

        Returns:
            Extraction result as a JSON-serializable dictionary.
        """
        url = f"{self.base_url}/extract_file"
        
        # Validate that the file exists
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Request parameters
        data = {
            'dataset_type': dataset_type,
            'use_vlm': str(use_vlm).lower(),
            'max_pages': str(max_pages)
        }
        
        print(f"Sending file '{file_path.name}' to Vision API (dataset_type={dataset_type}, use_vlm={use_vlm}, max_pages={max_pages})")
        
        # File to upload
        with open(file_path, 'rb') as file:
            files = {'file': (file_path.name, file, 'application/pdf')}
            
            try:
                response = self.session.post(url, data=data, files=files, timeout=600)
                response.raise_for_status()
                
                result = response.json()
                print("Vision API request succeeded")
                
                return result
                
            except requests.exceptions.RequestException as e:
                print(f"Vision API error: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"   Status: {e.response.status_code}")
                    print(f"   Response: {e.response.text[:500]}")
                return {"error": str(e), "status_code": getattr(e.response, 'status_code', None)}
    
    def health_check(self) -> Dict:
        """Check if the Vision API is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                return {"status": "healthy", "response": response.json()}
            else:
                return {"status": "unhealthy", "status_code": response.status_code}
        except Exception as e:
            return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    import sys

    client = VisionAPIClient()

    print(client.health_check())
    
    result_magnetic = client.extract_file(
        file_path="23_10.3390@cells8050444.pdf",
        dataset_type="magnetic",
        use_vlm=True,
        max_pages=0
    )

    print(result_magnetic)

    result_cytox = client.extract_file(
        file_path="23_10.3390@cells8050444.pdf",
        dataset_type="cytox",
        use_vlm=True,
        max_pages=0
    )
    print(result_cytox)

    result_nanozymes = client.extract_file(
        file_path="23_10.3390@cells8050444.pdf",
        dataset_type="nanozymes",
        use_vlm=True,
        max_pages=0
    )
    print(result_nanozymes)

    result_seltox = client.extract_file(
        file_path="23_10.3390@cells8050444.pdf",
        dataset_type="seltox",
        use_vlm=True,
        max_pages=0
    )
    print(result_seltox)

    result_synergy = client.extract_file(
        file_path="23_10.3390@cells8050444.pdf",
        dataset_type="synergy",
        use_vlm=True,
        max_pages=0
    )
    print(result_synergy)
