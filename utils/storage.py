"""
Storage Manager — Handles all local file system operations
===========================================================
Supports local disk, Google Colab Drive, and Kaggle datasets.

References
----------
Google Research (2017-2023). "Google Colaboratory." colab.research.google.com.
    Cloud notebook environment; COLAB_GPU env var detected for output path
    switching to /content/drive/MyDrive/.
Kaggle (2023). "Kaggle: ML competition and dataset platform." kaggle.com.
    KAGGLE_URL_BASE env var detected.
Python Software Foundation. "pathlib — Object-oriented filesystem paths."
    docs.python.org. Platform-independent path handling across Windows (local dev)
    and Linux (Colab/Kaggle).

Role
----
Abstracts file I/O across deployment environments. Provides: save_json(),
save_text(), save_binary(), ensure_dir(). Automatically routes output to Colab
Drive or Kaggle working directory when those environments are detected; defaults
to ~/Documents/Forensic_Reports/ on local desktop.
"""

from __future__ import annotations
import shutil
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from loguru import logger

from config import OUTPUT_DIR, IS_COLAB


class StorageManager:
    """Manages company-specific folder structures and file I/O."""

    def __init__(self, company_name: str, ticker: str = ""):
        self.company_name = self._sanitize(company_name)
        self.ticker = ticker.upper() if ticker else ""
        self.base_path = OUTPUT_DIR / self.company_name
        self._create_structure()

    def _sanitize(self, name: str) -> str:
        """Make safe folder name."""
        return "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()

    def _create_structure(self) -> None:
        """Create standard forensic investigation folder tree."""
        folders = [
            "Raw_Filings",
            "Parsed_Data",
            "Parsed_Data/Text",
            "Parsed_Data/Tables",
            "Financials",
            "Agent_Outputs",
            "Charts",
            "Reports",
            "Audit_Trail",
            "Final_Output",
            "Knowledge_Base",
        ]
        for folder in folders:
            (self.base_path / folder).mkdir(parents=True, exist_ok=True)
        logger.info(f"Storage ready at: {self.base_path}")

    # ─── Paths ─────────────────────────────────────────────
    @property
    def raw_filings(self) -> Path:
        return self.base_path / "Raw_Filings"

    @property
    def parsed_data(self) -> Path:
        return self.base_path / "Parsed_Data"

    @property
    def financials(self) -> Path:
        return self.base_path / "Financials"

    @property
    def agent_outputs(self) -> Path:
        return self.base_path / "Agent_Outputs"

    @property
    def charts(self) -> Path:
        return self.base_path / "Charts"

    @property
    def reports(self) -> Path:
        return self.base_path / "Reports"

    @property
    def audit_trail(self) -> Path:
        return self.base_path / "Audit_Trail"

    @property
    def final_output(self) -> Path:
        return self.base_path / "Final_Output"

    @property
    def knowledge_base(self) -> Path:
        return self.base_path / "Knowledge_Base"

    # ─── File Operations ────────────────────────────────────
    def save_json(self, data: Any, filename: str, subfolder: str = "Agent_Outputs") -> Path:
        path = self.base_path / subfolder / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        logger.debug(f"Saved JSON: {path}")
        return path

    def load_json(self, filename: str, subfolder: str = "Agent_Outputs") -> Any:
        path = self.base_path / subfolder / filename
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_text(self, text: str, filename: str, subfolder: str = "Parsed_Data/Text") -> Path:
        path = self.base_path / subfolder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def load_text(self, filename: str, subfolder: str = "Parsed_Data/Text") -> Optional[str]:
        path = self.base_path / subfolder / filename
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def save_binary(self, data: bytes, filename: str, subfolder: str = "Raw_Filings") -> Path:
        path = self.base_path / subfolder / filename
        with open(path, "wb") as f:
            f.write(data)
        return path

    def list_files(self, subfolder: str, extension: str = "*") -> list[Path]:
        folder = self.base_path / subfolder
        return sorted(folder.glob(f"*.{extension}"))

    def file_exists(self, filename: str, subfolder: str) -> bool:
        return (self.base_path / subfolder / filename).exists()

    def get_file_size_mb(self, filename: str, subfolder: str) -> float:
        path = self.base_path / subfolder / filename
        if not path.exists():
            return 0.0
        return path.stat().st_size / (1024 * 1024)


def get_storage(company_name: str, ticker: str = "") -> StorageManager:
    """Factory function to create a StorageManager."""
    return StorageManager(company_name, ticker)
