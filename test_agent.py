"""Quick test script for the game agent."""

import os
from dotenv import load_dotenv
from src.game_agent.agent import GameAgent

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("Error: OPENROUTER_API_KEY not found")
    exit(1)

print("Creating agent...")
agent = GameAgent(api_key=api_key)

print("Running test query...")
result = agent.run("Take a screenshot and describe what you see")

print("\n" + "="*60)
print("RESULT:")
print("="*60)
messages = result.get("messages", [])
for i, msg in enumerate(messages):
    print(f"\nMessage {i}: {type(msg).__name__}")
    if hasattr(msg, "content"):
        content = msg.content
        if isinstance(content, list):
            print(f"  Content (list with {len(content)} items)")
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image_url":
                        print(f"    - Image: {item['image_url']['url'][:50]}...")
                    else:
                        print(f"    - {item}")
        else:
            print(f"  Content: {str(content)[:200]}")
