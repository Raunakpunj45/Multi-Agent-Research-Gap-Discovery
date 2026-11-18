"""Silence one specific, harmless Streamlit file-watcher warning.

Streamlit's local-sources watcher (``streamlit.watcher.local_sources_watcher``)
calls ``get_module_paths()`` on EVERY module in ``sys.modules`` whenever the set
of loaded modules changes, probing ``__file__``, ``__spec__`` and ``__path__``.
Transformers registers lazy placeholder modules such as
``transformers.models.aria.image_processing_aria_fast``; probing one triggers
Transformers' lazy ``__getattr__``, which imports image-processing code that
needs ``torchvision``. This project only embeds text (BAAI/bge-small-en-v1.5)
and never uses those modules, so torchvision is not installed and every probe
logs a full ``ModuleNotFoundError`` traceback (103 of them after the embedding
model loads).

No Streamlit option prevents the probe: ``server.folderWatchBlacklist`` only
filters the paths AFTER the probe (and site-packages/venv are blacklisted by
default already), and ``server.fileWatcherType = "none"`` would disable hot
reload. The watcher itself keeps working unchanged; this filter only drops the
log record when ALL of these hold:

* it comes from the watcher's "Examining the path of %s raised" message;
* the module being examined is a ``transformers.`` module;
* the exception is ``ModuleNotFoundError`` for ``torchvision``.

Every other record (other modules, other missing packages, other exceptions)
still appears. The first suppressed record is announced once at INFO level.
"""
from __future__ import annotations

import logging

WATCHER_LOGGER = "streamlit.watcher.local_sources_watcher"
_MESSAGE_PREFIX = "Examining the path of"
_MISSING_MODULE = "torchvision"
_MODULE_PREFIX = "transformers."


class TransformersTorchvisionProbeFilter(logging.Filter):
    """Drop only the watcher's torchvision probe failures on transformers modules."""

    def __init__(self) -> None:
        super().__init__()
        self.suppressed = 0

    def matches(self, record: logging.LogRecord) -> bool:
        if not str(record.msg).startswith(_MESSAGE_PREFIX):
            return False
        module = record.args[0] if isinstance(record.args, tuple) and record.args else None
        if not isinstance(module, str) or not module.startswith(_MODULE_PREFIX):
            return False
        error = record.exc_info[1] if record.exc_info else None
        return (isinstance(error, ModuleNotFoundError)
                and (error.name or "").split(".")[0] == _MISSING_MODULE)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.matches(record):
            return True
        self.suppressed += 1
        if self.suppressed == 1:
            # Streamlit's logger factory, so the notice reaches the server
            # console at Streamlit's configured level (default: info).
            from streamlit.logger import get_logger

            get_logger(__name__).info(
                "Suppressing Streamlit file-watcher probes of transformers image-processing "
                "modules that need torchvision (not installed and not used by this app). "
                "The file watcher and hot reload are unaffected; other watcher warnings are "
                "still shown.")
        return False


def install() -> TransformersTorchvisionProbeFilter:
    """Attach the filter to the watcher logger once (idempotent across reruns)."""
    logger = logging.getLogger(WATCHER_LOGGER)
    for existing in logger.filters:
        if isinstance(existing, TransformersTorchvisionProbeFilter):
            return existing
    probe_filter = TransformersTorchvisionProbeFilter()
    logger.addFilter(probe_filter)
    return probe_filter
