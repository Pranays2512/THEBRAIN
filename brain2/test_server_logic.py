import sys
import os
import asyncio
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from faculties.whole_brain import WholeBrain

async def test():
    b = WholeBrain()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, b.sense, "What causes pain?")
    print(result)

asyncio.run(test())
