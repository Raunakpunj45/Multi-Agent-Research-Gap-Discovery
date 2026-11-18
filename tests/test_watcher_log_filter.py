"""Offline tests: only the watcher's torchvision probe noise is filtered.

Uses Streamlit's real ``get_module_paths`` on fake lazy modules that fail the way
Transformers' lazy image-processing placeholders do when torchvision is missing.
"""
import logging
import types
from pathlib import Path

import pytest
from streamlit.watcher import local_sources_watcher as lsw

from src.workflow import watcher_log_filter

ROOT = Path(__file__).resolve().parents[1]


class LazyModule(types.ModuleType):
    """Like transformers' lazy placeholders: attribute access triggers an import."""

    def __init__(self, name, error):
        super().__init__(name)
        self._error = error

    def __getattr__(self, attribute):
        raise self._error


def missing(package):
    return ModuleNotFoundError(f"No module named '{package}'", name=package)


@pytest.fixture
def installed():
    logger = logging.getLogger(watcher_log_filter.WATCHER_LOGGER)
    saved = list(logger.filters)
    logger.filters = [f for f in saved
                      if not isinstance(f, watcher_log_filter.TransformersTorchvisionProbeFilter)]
    probe_filter = watcher_log_filter.install()
    yield probe_filter
    logger.filters = saved


def watcher_warnings(caplog, module):
    caplog.clear()
    with caplog.at_level(logging.INFO):
        paths = lsw.get_module_paths(module)
    warnings = [r for r in caplog.records
                if r.name == watcher_log_filter.WATCHER_LOGGER and r.levelno >= logging.WARNING]
    return paths, warnings


def test_without_the_filter_the_probe_logs_a_torchvision_traceback(caplog):
    logger = logging.getLogger(watcher_log_filter.WATCHER_LOGGER)
    saved = list(logger.filters)
    logger.filters = []  # other tests (or the app) may have installed the filter
    try:
        module = LazyModule("transformers.models.aria.image_processing_aria_fast", missing("torchvision"))
        _, warnings = watcher_warnings(caplog, module)
    finally:
        logger.filters = saved
    assert warnings and "Examining the path of" in warnings[0].getMessage()
    assert isinstance(warnings[0].exc_info[1], ModuleNotFoundError)


class Collect(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_transformers_torchvision_probe_is_suppressed_once_announced(installed, caplog):
    from streamlit.logger import get_logger

    notice_logger = get_logger(watcher_log_filter.__name__)   # Streamlit's console logger
    notices = Collect()
    notice_logger.addHandler(notices)
    try:
        module = LazyModule("transformers.models.beit.image_processing_beit_fast",
                            missing("torchvision"))
        paths, warnings = watcher_warnings(caplog, module)
        assert warnings == [] and paths == set()
        assert installed.suppressed >= 1
        assert len(notices.records) == 1 and notices.records[0].levelno == logging.INFO
        assert "hot reload are unaffected" in notices.records[0].getMessage()

        other = LazyModule("transformers.models.vit.image_processing_vit_fast",
                           missing("torchvision.transforms"))
        _, warnings = watcher_warnings(caplog, other)
        assert warnings == []
        assert len(notices.records) == 1                       # announced only once
    finally:
        notice_logger.removeHandler(notices)


@pytest.mark.parametrize("name, error", [
    ("mypackage.vision", missing("torchvision")),                      # not transformers
    ("transformers.models.aria.image_processing_aria_fast", missing("PIL")),  # other package
    ("transformers.models.bert.modeling_bert", RuntimeError("real failure")),  # other error
    ("src.workflow.demo_broken", missing("torchvision")),              # project code
])
def test_every_other_watcher_warning_is_still_shown(installed, caplog, name, error):
    _, warnings = watcher_warnings(caplog, LazyModule(name, error))
    assert warnings, f"warning for {name} must not be hidden"
    assert warnings[0].exc_info[1] is error


def test_other_loggers_are_untouched(installed, caplog):
    with caplog.at_level(logging.WARNING):
        logging.getLogger("streamlit.other").warning(
            "Examining the path of %s raised:", "transformers.x", exc_info=missing("torchvision"))
    assert len(caplog.records) == 1


def test_install_is_idempotent_across_reruns(installed):
    logger = logging.getLogger(watcher_log_filter.WATCHER_LOGGER)
    assert watcher_log_filter.install() is installed
    assert sum(isinstance(f, watcher_log_filter.TransformersTorchvisionProbeFilter)
               for f in logger.filters) == 1


def test_watcher_still_finds_real_module_paths(installed):
    paths = lsw.get_module_paths(watcher_log_filter)
    assert str(Path(watcher_log_filter.__file__).resolve()) in {
        str(Path(p).resolve()) for p in paths}


def test_no_deprecated_use_container_width_in_project_code():
    offenders = [
        str(path.relative_to(ROOT)) for path in ROOT.rglob("*.py")
        if not {"venv", ".venv", "__pycache__"} & set(path.relative_to(ROOT).parts)
        and path.name != Path(__file__).name
        and "use_container_width" in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == []
