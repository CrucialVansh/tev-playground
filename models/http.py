import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict


def post_json(url: str, api_key: str, payload: Dict[str, Any], timeout: float = 60) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:400]
            last_error = RuntimeError(f"HTTP {error.code}: {detail}")
            if error.code not in {429, 500, 502, 503, 504} or attempt == 1:
                raise last_error from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt == 1:
                raise
        time.sleep(1)
    raise RuntimeError(f"Request failed: {last_error}")
