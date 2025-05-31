import argparse

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server import Server
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route

from game.bomb import Bomb
from game.modules.module import ActionResult

# Initialize FastMCP server
mcp = FastMCP("Game")
bomb = Bomb()

BOMB_EXPLODED = f"=== BOOM! THE BOMB HAS EXPLODED. GAME OVER. === \n\n'"
BOMB_DISARMED = f"=== BOMB SUCCESSFULLY DISARMED! CONGRATULATIONS! ===\n\n"
UNKNOWN_COMMAND = "Unknown command. Type 'help' for available commands.\n\n"
HELP_TEXT = """Keep Talking and Nobody Explodes

Game Description:
In this game, there are two players:
• The Defuser (you): Sees the bomb's current module 'screen' but not the manual.
• The Manual Expert: Has access to the defusal manual but not the bomb's display.
Players must collaborate by exchanging text instructions to solve each module.
The goal is to disarm all modules before time runs out or the bomb explodes.
Each module has specific rules that must be followed precisely.
Communication is key - the defuser must clearly describe what they see,
and the expert must provide clear instructions based on the manual.

"""


@mcp.tool()
async def game_interaction(command: str) -> str:
    """Get the current status of the game.

    Args:
        command: str: The command to execute.
    """
    print(f"Received command: {command}")
    if command == "help":
        return HELP_TEXT

    elif command == "state":
        res = f"=== BOMB STATE ===\n\n"

        state, actions = bomb.state()
        res += state + "\n"
        if actions:
            res += "\nAvailable commands:" + "\n"
            for action in actions:
                res += f"  {action}" + "\n"
        res += "\n"

        return res

    elif command.startswith(("cut", "press", "hold", "release")):
        result = bomb.do_action(command)

        if result == ActionResult.CHANGED:
            res = "The module state has changed." + "\n"
            state, actions = bomb.state()
            res += "\nCurrent state:" + "\n"
            res += state
            if actions:
                res += "\nAvailable commands:" + "\n"
                for action in actions:
                    res += f"  {action}" + "\n"
            res += "\n"

            return res

        elif result == ActionResult.DISARMED:
            return BOMB_DISARMED
        elif result == ActionResult.EXPLODED:
            return BOMB_EXPLODED

    return UNKNOWN_COMMAND


@mcp.tool()
async def get_manual() -> str:
    """Get the manual for the game."""
    if bomb.exploded:
        return BOMB_EXPLODED
    if bomb.disarmed:
        return BOMB_DISARMED

    return bomb.modules[bomb.current_module].instruction()


@mcp.tool()
async def reset(module: str):
    global bomb
    bomb = Bomb(module)
    return 'Game resetted'


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provied mcp server with SSE."""
    sse = SseServerTransport("/session_id/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/", endpoint=handle_sse),
            Mount("/session_id/", app=sse.handle_post_message),
        ],
    )


if __name__ == "__main__":
    mcp_server = mcp._mcp_server  # noqa: WPS437

    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    args = parser.parse_args()

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)
