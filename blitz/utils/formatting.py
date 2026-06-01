"""
Subtitle formatting utilities for SRT and VTT formats.

Provides functions to generate subtitle files from timestamp segments.
"""

import datetime
import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


def format_srt_time(seconds: float) -> str:
    """
    Convert seconds to SRT time format: HH:MM:SS,mmm

    Args:
        seconds: Time in seconds (can be float)

    Returns:
        SRT formatted time string (e.g., "00:01:23,456")

    Example:
        >>> format_srt_time(83.456)
        '00:01:23,456'
    """
    # Ensure non-negative value
    sanitized_total_seconds = max(0.0, seconds)

    delta = datetime.timedelta(seconds=sanitized_total_seconds)
    total_int_seconds = int(delta.total_seconds())

    hours = total_int_seconds // 3600
    remainder = total_int_seconds % 3600
    minutes = remainder // 60
    seconds_part = remainder % 60
    milliseconds = delta.microseconds // 1000

    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"


def format_vtt_time(seconds: float) -> str:
    """
    Convert seconds to WebVTT time format: HH:MM:SS.mmm

    Args:
        seconds: Time in seconds (can be float)

    Returns:
        VTT formatted time string (e.g., "00:01:23.456")

    Note:
        VTT uses dot (.) instead of comma (,) for milliseconds

    Example:
        >>> format_vtt_time(83.456)
        '00:01:23.456'
    """
    # Reuse SRT logic but replace comma with dot
    srt_time = format_srt_time(seconds)
    return srt_time.replace(",", ".")


def generate_srt_content(
    segment_timestamps: List[Union[Dict[str, Any], List[Any], tuple]],
) -> str:
    """
    Generate SRT formatted string from segment timestamps.

    Args:
        segment_timestamps: List of timestamp segments in one of these formats:
            - Dict format: {'start': 0.0, 'end': 2.5, 'segment': 'Hello'}
            - List/Tuple format: [0.0, 2.5, 'Hello']

    Returns:
        SRT formatted string ready to write to .srt file

    Example:
        >>> segments = [
        ...     {'start': 0.0, 'end': 2.5, 'segment': 'Hello world'},
        ...     {'start': 2.5, 'end': 5.0, 'segment': 'How are you?'}
        ... ]
        >>> print(generate_srt_content(segments))
        1
        00:00:00,000 --> 00:00:02,500
        Hello world

        2
        00:00:02,500 --> 00:00:05,000
        How are you?
        <BLANKLINE>
    """
    srt_content = []

    for i, ts in enumerate(segment_timestamps):
        try:
            # Handle both dict and list/tuple formats
            if isinstance(ts, dict):
                start_time = format_srt_time(ts.get("start", 0))
                end_time = format_srt_time(ts.get("end", 0))
                text = ts.get("segment") or ts.get("text") or ""

            elif isinstance(ts, (list, tuple)) and len(ts) >= 3:
                # Handle [start, end, text] format
                start_time = format_srt_time(ts[0])
                end_time = format_srt_time(ts[1])
                text = ts[2]

            else:
                logger.warning("Unexpected timestamp format at index %d: %r", i, ts)
                continue

            # Build SRT entry
            srt_content.append(str(i + 1))  # Sequence number
            srt_content.append(f"{start_time} --> {end_time}")  # Timestamp line
            srt_content.append(text)  # Subtitle text
            srt_content.append("")  # Blank line separator

        except Exception as e:
            logger.error("Error processing timestamp %d: %s, data: %r", i, e, ts)
            continue

    return "\n".join(srt_content)


def generate_vtt_content(
    segment_timestamps: List[Union[Dict[str, Any], List[Any], tuple]],
) -> str:
    """
    Generate WebVTT formatted string from segment timestamps.

    Args:
        segment_timestamps: List of timestamp segments in one of these formats:
            - Dict format: {'start': 0.0, 'end': 2.5, 'segment': 'Hello'}
            - List/Tuple format: [0.0, 2.5, 'Hello']

    Returns:
        WebVTT formatted string ready to write to .vtt file

    Example:
        >>> segments = [
        ...     {'start': 0.0, 'end': 2.5, 'segment': 'Hello world'},
        ...     {'start': 2.5, 'end': 5.0, 'segment': 'How are you?'}
        ... ]
        >>> print(generate_vtt_content(segments))
        WEBVTT
        <BLANKLINE>
        1
        00:00:00.000 --> 00:00:02.500
        Hello world
        <BLANKLINE>
        2
        00:00:02.500 --> 00:00:05.000
        How are you?
        <BLANKLINE>
    """
    vtt_content = ["WEBVTT", ""]  # WebVTT header with blank line

    for i, ts in enumerate(segment_timestamps):
        try:
            # Handle both dict and list/tuple formats
            if isinstance(ts, dict):
                start_time = format_vtt_time(ts.get("start", 0))
                end_time = format_vtt_time(ts.get("end", 0))
                text = ts.get("segment") or ts.get("text") or ""

            elif isinstance(ts, (list, tuple)) and len(ts) >= 3:
                # Handle [start, end, text] format
                start_time = format_vtt_time(ts[0])
                end_time = format_vtt_time(ts[1])
                text = ts[2]

            else:
                logger.warning("Unexpected timestamp format at index %d: %r", i, ts)
                continue

            # Build VTT entry (optional cue identifier)
            vtt_content.append(str(i + 1))  # Cue identifier
            vtt_content.append(f"{start_time} --> {end_time}")  # Timestamp line
            vtt_content.append(text)  # Subtitle text
            vtt_content.append("")  # Blank line separator

        except Exception as e:
            logger.error("Error processing timestamp %d: %s, data: %r", i, e, ts)
            continue

    return "\n".join(vtt_content)
