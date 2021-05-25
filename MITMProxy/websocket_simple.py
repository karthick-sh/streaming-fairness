"""Process individual messages from a WebSocket connection."""
import re
from mitmproxy import ctx
import json

def parse_websocket_message(content):
    numOpens = 0
    parsed = ""
    for c in content:
        if c == "{":
            numOpens += 1
        elif c == "}":
            numOpens -= 1
        
        parsed += c

        if numOpens == 0:
            break
    print("Parsed: ", parsed)
    return json.loads(parsed)

def websocket_message(flow):
    # get the latest message
    message = flow.messages[-1]

    # was the message sent from the client or server?
    if message.from_client:
        parsed = json.loads(message.content)
        print("[CLIENT MSG] -> {}".format(parsed))
    # else:
    #     parsed = parse_websocket_message(message.content[2:300].decode('utf-8','ignore'))
    #     print("[WEBSOCKET] -> {}".format(parsed))

    # # manipulate the message content
    # message.content = re.sub(r'^Hello', 'HAPPY', message.content)

    # if 'FOOBAR' in message.content:
    #     # kill the message and not send it to the other endpoint
    #     message.kill()