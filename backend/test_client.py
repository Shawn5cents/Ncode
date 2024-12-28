import asyncio
import json
import websockets

async def test_generation():
    client_id = "test_client"
    uri = f"ws://localhost:8000/ws/{client_id}"
    print(f"Connecting to {uri}...")
    try:
        # Add retry logic with exponential backoff
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                async with websockets.connect(
                    uri,
                    subprotocols=["json"],
                    open_timeout=60,  # Further increase connection timeout
                    close_timeout=20,
                    ping_interval=30,
                    ping_timeout=30
                ) as websocket:
                    print(f"Connected on attempt {attempt + 1}")
                    break
            except (websockets.exceptions.WebSocketException, TimeoutError) as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Connection failed, retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                    # Manually send Origin header after connection
                    await websocket.send(json.dumps({
                        "type": "set_origin",
                        "origin": "http://localhost:8000"
                    }))
                    print("Connected! Sending generation request...")

                    # Send generation request
                    await websocket.send(json.dumps({
                        "type": "prompt_request",
                        "client_id": client_id,
                        "prompt": "Create a simple hello world program in Python",
                        "model": "coder",
                        "temperature": 0.7,
                        "max_tokens": 100
                    }))
                    print("Request sent, waiting for response...")

                    # Receive and print responses
                    while True:
                        try:
                            response = await websocket.recv()
                            data = json.loads(response)
                            print(f"\nReceived message type: {data['type']}")

                            if data["type"] == "status":
                                print("Status:", data.get("message", ""))
                            elif data["type"] == "planning_stream":
                                print("Planning:", data.get("content", ""))
                            elif data["type"] == "coding_stream":
                                print("Coding:", data.get("content", ""))
                            elif data["type"] == "complete":
                                print("\nGeneration complete!")
                                break
                            elif data["type"] == "error":
                                print("\nError:", data.get("message", ""))
                                break
                            else:
                                print("Other message:", data)

                        except websockets.exceptions.ConnectionClosed:
                            print("\nConnection closed")
                            break
                        except json.JSONDecodeError as e:
                            print(f"\nFailed to parse response: {response}")
                            print(f"Error: {e}")
                            break
    except websockets.exceptions.WebSocketException as e:
        print(f"WebSocket error: {e}")
    except Exception as e:
        print(f"Unexpected error:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_generation())
