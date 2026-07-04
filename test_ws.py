import asyncio
import websockets
import json

async def test():
    async with websockets.connect("ws://127.0.0.1:7860/ws") as websocket:
        lobby_msg = await websocket.recv()
        print("Received:", lobby_msg)
        
        await websocket.send(json.dumps({"type": "start_game", "bot_count": 2}))
        
        state_msg = await websocket.recv()
        print("Received State length:", len(state_msg))
        data = json.loads(state_msg)
        if data.get("type") == "state_update":
            print("Players:", len(data["data"]["players"]))
        
asyncio.run(test())
