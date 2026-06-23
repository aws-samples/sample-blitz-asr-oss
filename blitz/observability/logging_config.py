import logging, json, sys
from datetime import datetime, timezone
from contextvars import ContextVar

_request_id_var = ContextVar("request_id", default=None)

def set_request_context(request_id): _request_id_var.set(request_id)
def clear_request_context(): _request_id_var.set(None)

class JSONFormatter(logging.Formatter):
    def format(self, record):
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "level": record.levelname, "logger": record.name, "message": record.getMessage(), "service": "blitz-asr"}
        req_id = _request_id_var.get()
        if req_id: entry["request_id"] = req_id
        if record.exc_info: entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)

def setup_logging(log_level="INFO", json_format=True):
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper()))
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter() if json_format else logging.Formatter("%(asctime)s [%(levelname)s] %(name)s | %(message)s"))
    root.addHandler(handler)
    for lib in ["nemo", "transformers", "speechbrain", "torch"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
