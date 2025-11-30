"""Debug script to see what the tool returns."""

from src.game_agent.tools import take_screenshot

print("Testing take_screenshot tool...")
result = take_screenshot.invoke({})

print(f"\nResult type: {type(result)}")
print(f"\nResult: {result}")
