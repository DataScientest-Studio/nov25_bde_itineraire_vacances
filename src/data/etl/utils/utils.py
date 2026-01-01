import time
import requests

def download_with_retry(url: str, retries: int = 3, timeout: int = 30) -> bytes:
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)