import atexit
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from hydra.core.hydra_config import HydraConfig


class TeeStream:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self.encoding = getattr(stream, "encoding", "utf-8")

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.stream.isatty()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "run"


def get_run_dir(cfg, dataset_name):
    try:
        hydra_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        hydra_dir = None

    subdir = cfg.get("subdir")
    subdir = str(subdir).strip() if subdir is not None else ""
    if subdir and subdir.lower() not in {"none", "null"} and not subdir.startswith("${"):
        run_dir = Path("outputs", safe_name(subdir))
    elif hydra_dir is not None:
        run_dir = hydra_dir
    else:
        stem = Path(str(dataset_name)).with_suffix("").as_posix()
        run_id = safe_name(
            f"{datetime.now():%Y%m%d_%H%M%S_%f}_{cfg.output_model_name}_{stem}_{os.getpid()}"
        )
        run_dir = Path("outputs", run_id)

    run_dir = run_dir.resolve()
    return run_dir, safe_name(run_dir.name)


def tee_output_to_file(run_dir, dataset_name):
    run_dir.mkdir(parents=True, exist_ok=True)
    log_stem = safe_name(Path(str(dataset_name)).with_suffix("").name)
    log_path = run_dir / f"{log_stem}.log"
    log_file = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
    stdout, stderr = sys.stdout, sys.stderr
    print(f"Logging stdout/stderr to {log_path}")
    sys.stdout = TeeStream(stdout, log_file)
    sys.stderr = TeeStream(stderr, log_file)

    py_handler = logging.FileHandler(log_path, encoding="utf-8")
    py_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(py_handler)

    loguru_logger = None
    loguru_sink_id = None
    try:
        from loguru import logger as loguru_logger

        loguru_sink_id = loguru_logger.add(log_file, level="DEBUG", colorize=False)
    except Exception:
        pass

    def close_log():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            if loguru_logger is not None and loguru_sink_id is not None:
                loguru_logger.remove(loguru_sink_id)
            logging.getLogger().removeHandler(py_handler)
            py_handler.close()
            if isinstance(sys.stdout, TeeStream) and sys.stdout.log_file is log_file:
                sys.stdout = stdout
            if isinstance(sys.stderr, TeeStream) and sys.stderr.log_file is log_file:
                sys.stderr = stderr
            log_file.close()

    atexit.register(close_log)
    return log_path
