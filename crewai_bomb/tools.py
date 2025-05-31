import asyncio
from typing import Optional

from crewai.tools import BaseTool

from game_mcp.game_client import Defuser, Expert

from pydantic import ConfigDict

# Feel free to import any libraries you need - if needed change requirements.txt
# In this file it also applies to classes and functions :)


class DefuserTool(BaseTool):
    name: str = "defuser_tool"
    description: str = "Use this tool to interact with the bomb. THIS TOOL CANNOT BE USED TO COMMUNICATE WITH EXPERT!"
    defuser: Optional[Defuser] = None

    def __init__(self, server_url: str):
        super().__init__()
        self.defuser = Defuser()
        asyncio.run(self.defuser.connect_to_server(server_url))

    def _run(self, command: str) -> str:
        """Sync wrapper to run defuser action"""
        return asyncio.run(self.defuser.run(command))

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ExpertTool(BaseTool):
    name: str = "expert_tool"
    description: str = ("Use this tool to retrieve the manual as the expert. THIS TOOL CANNOT BE USED TO COMMUNICATE "
                        "WITH DEFUSER!")
    expert: Optional[Expert] = None

    def __init__(self, server_url: str):
        super().__init__()
        self.expert = Expert()
        asyncio.run(self.expert.connect_to_server(server_url))

    def _run(self, _: str = "") -> str:
        """Sync wrapper to run expert action"""
        return asyncio.run(self.expert.run())

    model_config = ConfigDict(arbitrary_types_allowed=True)


class UppercaseVowelTool(BaseTool):
    name: str = "uppercase_vowel_tool"
    description: str = "Checks if a given bomb ID contains any uppercase vowels (A, E, I, O, U)."

    def _run(self, input_str: str) -> str:
        has_uppercase_vowels = any(char in "AEIOU" for char in input_str)
        return str(has_uppercase_vowels)

    model_config = ConfigDict(arbitrary_types_allowed=True)


# Helper for simon solver
def parse_flash_table(table_str: str) -> dict:
    lines = [line.strip() for line in table_str.strip().split('\n') if line.strip()]

    # Extract header (rounds)
    header = lines[0].split('|')[2:]  # Skip the first two columns (empty and label)
    header = [h.strip() for h in header]

    table = {}

    for row in lines[2:]:  # Skip header and separator
        columns = row.split('|')
        color = columns[1].strip()
        responses = [c.strip() for c in columns[2:]]
        table[color] = dict(zip(header, responses))

    return table


class SimonSolverTool(BaseTool):
    name: str = "simon_solver_tool"
    description: str = ("""
        INPUT:
        table_str: string containing appriopriate table from manual.
        colours: string containing colours which flashed separated by ,
        OUTPUT:
        String containing colours of buttons that should be pressed.
    """)

    def _run(self, table_str: str, colours: str) -> str:
        print(table_str)
        print(colours)
        table = parse_flash_table(table_str)

        input_colours = [c.strip() for c in colours.split(',')]

        responses = []
        for i, color in enumerate(input_colours):
            round_key = f"Round {i + 1}"
            if color in table and round_key in table[color]:
                responses.append(table[color][round_key])
            else:
                responses.append("Invalid")

        return ', '.join(responses)

    model_config = ConfigDict(arbitrary_types_allowed=True)
