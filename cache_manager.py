"""
cache_manager.py — IP8 Patent Infringement Detection System
Pickle-based local file cache keyed by patent number. Saves/loads intermediate
pipeline results to avoid re-running expensive LLM and scraping operations.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Any, Optional

from config import CACHE_DIRECTORY, CACHE_FILE_EXTENSION

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================


def _build_cache_file_path(patent_number: str, step_key: str) -> Path:
    """
    Construct the full path to a cache file for a given patent number and step.

    Args:
        patent_number: The patent identifier (e.g. 'US10696212B2').
        step_key: A short label identifying which pipeline step this cache belongs to.

    Returns:
        A resolved Path object pointing to the cache file.
    """
    sanitized_patent = patent_number.strip().replace("/", "_").replace("\\", "_")
    filename = f"{sanitized_patent}__{step_key}{CACHE_FILE_EXTENSION}"
    return Path(CACHE_DIRECTORY) / filename


def _ensure_cache_directory_exists() -> None:
    """Create the .ip8_cache directory if it does not already exist."""
    Path(CACHE_DIRECTORY).mkdir(parents=True, exist_ok=True)


# =============================================================================
# PUBLIC API
# =============================================================================


def save_to_cache(patent_number: str, step_key: str, data: Any) -> None:
    """
    Serialize and persist pipeline data for a patent step to a pickle file.

    Args:
        patent_number: The patent identifier used as the cache key.
        step_key: Label for the pipeline step (e.g., 'patent_data', 'claims').
        data: Any Python-serializable object to cache.
    """
    _ensure_cache_directory_exists()
    cache_path = _build_cache_file_path(patent_number, step_key)
    try:
        with open(cache_path, "wb") as cache_file:
            pickle.dump(data, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Cache saved: %s", cache_path)
    except (OSError, pickle.PicklingError) as exc:
        logger.warning(
            "Failed to save cache for %s / %s: %s", patent_number, step_key, exc
        )


def load_from_cache(patent_number: str, step_key: str) -> Optional[Any]:
    """
    Load previously cached pipeline data for a patent step if available.

    Args:
        patent_number: The patent identifier used as the cache key.
        step_key: Label for the pipeline step.

    Returns:
        The deserialized data object, or None if no cache exists.
    """
    cache_path = _build_cache_file_path(patent_number, step_key)
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as cache_file:
            data = pickle.load(cache_file)
        logger.info("Cache loaded: %s", cache_path)
        return data
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        logger.warning(
            "Failed to load cache for %s / %s: %s", patent_number, step_key, exc
        )
        return None


def cache_exists(patent_number: str, step_key: str) -> bool:
    """
    Check whether a cache file exists for a given patent and step.

    Args:
        patent_number: The patent identifier.
        step_key: Label for the pipeline step.

    Returns:
        True if the cache file exists and is non-empty, False otherwise.
    """
    cache_path = _build_cache_file_path(patent_number, step_key)
    return cache_path.exists() and cache_path.stat().st_size > 0


def clear_cache_for_patent(patent_number: str) -> int:
    """
    Delete all cache files associated with a specific patent number.

    Args:
        patent_number: The patent identifier whose cache should be cleared.

    Returns:
        The number of files successfully deleted.
    """
    cache_dir = Path(CACHE_DIRECTORY)
    if not cache_dir.exists():
        return 0

    sanitized_patent = patent_number.strip().replace("/", "_").replace("\\", "_")
    deleted_count = 0
    for cache_file in cache_dir.glob(f"{sanitized_patent}__*{CACHE_FILE_EXTENSION}"):
        try:
            cache_file.unlink()
            deleted_count += 1
            logger.info("Deleted cache file: %s", cache_file)
        except OSError as exc:
            logger.warning("Failed to delete cache file %s: %s", cache_file, exc)

    return deleted_count


def list_cached_patents() -> list[str]:
    """
    Return a list of unique patent numbers that have at least one cache file.

    Returns:
        A sorted list of patent number strings found in the cache directory.
    """
    cache_dir = Path(CACHE_DIRECTORY)
    if not cache_dir.exists():
        return []

    patent_numbers: set[str] = set()
    for cache_file in cache_dir.glob(f"*{CACHE_FILE_EXTENSION}"):
        stem = cache_file.stem  # filename without extension
        if "__" in stem:
            patent_part = stem.split("__")[0]
            patent_numbers.add(patent_part)

    return sorted(patent_numbers)


def get_cache_size_bytes(patent_number: str) -> int:
    """
    Calculate the total disk size of all cache files for a patent.

    Args:
        patent_number: The patent identifier.

    Returns:
        Total size in bytes across all cache files for this patent.
    """
    cache_dir = Path(CACHE_DIRECTORY)
    if not cache_dir.exists():
        return 0

    sanitized_patent = patent_number.strip().replace("/", "_").replace("\\", "_")
    total_bytes = 0
    for cache_file in cache_dir.glob(f"{sanitized_patent}__*{CACHE_FILE_EXTENSION}"):
        try:
            total_bytes += cache_file.stat().st_size
        except OSError:
            pass

    return total_bytes
