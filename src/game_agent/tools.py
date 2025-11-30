"""Tools for the game agent."""

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import mss
from langchain_core.tools import tool
from PIL import Image


@tool
def take_screenshot(save_path: Optional[str] = None) -> dict:
    """Take a screenshot of the entire screen.

    Args:
        save_path: Optional path to save the screenshot. If not provided,
                   saves to screenshots folder with timestamp.

    Returns:
        dict with 'path' (str) and 'base64' (str) of the screenshot.
    """
    # Create screenshots directory if it doesn't exist
    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)

    # Generate filename with timestamp if not provided
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = str(screenshots_dir / f"screenshot_{timestamp}.png")

    # Take screenshot
    with mss.mss() as sct:
        # Capture the primary monitor
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)

        # Convert to PIL Image
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

        # Save the image
        img.save(save_path)

        # Convert to base64 for Claude vision
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return {
        "path": save_path,
        "base64": img_base64,
        "message": f"Screenshot saved to {save_path}"
    }


@tool
def take_region_screenshot(x: int, y: int, width: int, height: int,
                          save_path: Optional[str] = None) -> dict:
    """Take a screenshot of a specific region of the screen.

    Args:
        x: X coordinate of the top-left corner
        y: Y coordinate of the top-left corner
        width: Width of the region
        height: Height of the region
        save_path: Optional path to save the screenshot

    Returns:
        dict with 'path' (str) and 'base64' (str) of the screenshot.
    """
    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = str(screenshots_dir / f"screenshot_region_{timestamp}.png")

    with mss.mss() as sct:
        # Define the region
        monitor = {"top": y, "left": x, "width": width, "height": height}
        screenshot = sct.grab(monitor)

        # Convert to PIL Image
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

        # Save the image
        img.save(save_path)

        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return {
        "path": save_path,
        "base64": img_base64,
        "message": f"Screenshot of region saved to {save_path}"
    }
