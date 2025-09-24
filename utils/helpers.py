from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout

# --- Timeouts (small, invisible safety net) ---
RETRIEVAL_TIMEOUT_S = 12
LLM_TIMEOUT_S = 25

def _attempt_with_timeout(fn, timeout_s: float, retries: int = 1):
    """
    Run fn() with a timeout. If it times out or errors, retry up to `retries` once.
    Returns (ok: bool, value_or_none, err_msg_or_none).
    """
    last_err = None
    for _ in range(retries + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(fn)
                return True, fut.result(timeout=timeout_s), None
        except _FutTimeout:
            last_err = f"timed out after {timeout_s}s"
        except Exception as e:
            last_err = str(e)
    return False, None, last_err