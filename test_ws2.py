import asyncio
import websockets
import json

async def test():
    async with websockets.connect("wss://ptanay-coup-rl-backend.hf.space/ws") as websocket:
        print("Connected")
        while True:
            try:
                msg = await websocket.recv()
                print("Received:", msg)
                if json.loads(msg).get("type") == "lobby_state":
                    await websocket.send(json.dumps({"type": "start_game", "bot_count": 3}))
                    print("Sent start_game")
                if json.loads(msg).get("type") == "state_update":
                    print("State update received, successful start")
                    break
            except Exception as e:
                print("Error:", e)
                break

asyncio.run(test())
