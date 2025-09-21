import requests
import time
from pathlib import Path
from typing import Dict, List


class NERAPIClient:
    """Client for interacting with the NER API"""
    
    def __init__(self, base_url: str = "http://77.234.216.102:17629"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def extract_entities(
        self,
        file_path: str,
        extraction_type: str = "cytox",
        model_names: List[str] = None,
        max_pages: int = 0
    ) -> Dict:
        """
        Extract named entities from a PDF file using the NER API.

        Args:
            file_path: Path to the PDF file.
            extraction_type: Extraction type (e.g., "cytox", "nanozymes").
            model_names: List of model names to use. If None, defaults are used.
            max_pages: Maximum number of pages to process (0 = all).

        Returns:
            Extraction result as a JSON-serializable dictionary.
        """
        if model_names is None:
            model_names = [
                "zjkarina/nanoMINER_sft-Llama-3.1-8B-unsloth-full",
                "zjkarina/nanoMINER_sft-Mistral-7B-Instruct-v0.2-full"
            ]
        
        url = f"{self.base_url}/upload"
        
        # Validate that the file exists
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Request parameters
        data = {
            'use_vlm': 'false',
            'dataset_type': extraction_type,
            'max_pages': str(max_pages)
        }
        
        # Add models
        for model in model_names:
            data['model_name'] = model
        
        print(f"Sending file '{file_path.name}' to NER API (extraction_type={extraction_type}, models={len(model_names)}, max_pages={max_pages})")
        
        # File to upload
        with open(file_path, 'rb') as file:
            files = {'file': (file_path.name, file, 'application/pdf')}
            
            try:
                response = self.session.post(url, data=data, files=files, timeout=900)
                response.raise_for_status()
                
                result = response.json()
                print("NER API request succeeded")
                
                return result
                
            except requests.exceptions.RequestException as e:
                print(f"NER API error: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"   Status: {e.response.status_code}")
                    print(f"   Response: {e.response.text[:500]}")
                return {"error": str(e), "status_code": getattr(e.response, 'status_code', None)}
    
    def get_task_status(self, task_id: str) -> Dict:
        """Get the status of a NER task by task ID."""
        url = f"{self.base_url}/status/{task_id}"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status_code": getattr(e.response, 'status_code', None)}
    
    def wait_for_completion(self, task_id: str, check_interval: int = 30, max_wait: int = 3600) -> Dict:
        """
        Wait for a NER task to complete and return the final status payload.

        Args:
            task_id: Task identifier returned by the NER API.
            check_interval: Polling interval in seconds.
            max_wait: Maximum time to wait in seconds.
        """
        start_time = time.time()
        print(f"Waiting for task {task_id} to complete (timeout={max_wait//60} min, interval={check_interval}s)")
        
        while time.time() - start_time < max_wait:
            status_result = self.get_task_status(task_id)
            if status_result.get("error"):
                print(f"Status check error: {status_result['error']}")
                return status_result
            
            current_status = status_result.get('status', 'unknown')
            # Completed successfully and data available
            if 'results' in status_result and 'data' in status_result['results']:
                print(f"Task {task_id} completed successfully")
                return status_result
            # Failed terminal states
            if current_status in ['failed', 'error']:
                print(f"Task {task_id} failed with status '{current_status}'")
                return status_result
            
            time.sleep(check_interval)
        
        return {"error": "Timeout waiting for task", "task_id": task_id}
    
    def health_check(self) -> Dict:
        """Check if the NER API is healthy"""
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
    client = NERAPIClient()
    print(client.health_check())
    result_cytox = client.extract_entities(
        file_path="23_10.3390@cells8050444.pdf",
        extraction_type="cytox",
        model_names=None,
        max_pages=0
    )
    task_id = result_cytox.get('task_id') or result_cytox.get('id') if isinstance(result_cytox, dict) else None
    if task_id:
        result_cytox = client.wait_for_completion(task_id, check_interval=45, max_wait=5400)
    print(result_cytox)
