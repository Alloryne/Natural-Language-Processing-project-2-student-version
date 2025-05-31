from crewai import Agent, Crew, Task, LLM

from tools import DefuserTool, ExpertTool, UppercaseVowelTool, SimonSolverTool

server_url = "http://127.0.0.1:8080"

# Models:
# gpt-3.5-turbo
# ollama/qwen3:14b
openai_model = LLM(
    model="openai/gpt-3.5-turbo",
    temperature=0.3,
    max_tokens=300,
    max_iter=10
)

qwen_model = LLM(
    model="ollama/qwen3:14B",
)

groq_model = LLM(
    model="groq/qwen-qwq-32b",
    temperature=0.3
)

USED_MODEL = qwen_model

# Create tools
defuser_tool = DefuserTool(server_url)
expert_tool = ExpertTool(server_url)
uppercase_vowel_tool = UppercaseVowelTool()
simon_solver_tool = SimonSolverTool()

# Agents
defuser_agent = Agent(
    role="Bomb Defuser",
    goal="Describe bomb modules and execute expert instructions.",
    backstory=
    """
        You're playing a game where you get descriptions of a bomb and you have to defuse it
        by doing actions using based on advice from an expert with a manual.
        The expert gives you actions to execute.
        USE ONLY ACTIONS LIKE THOSE LISTED AFTER LINE 'Available commands:' 
        in bomb state.
    """,
    llm=USED_MODEL,
    tools=[defuser_tool],
    verbose=True,
    cache=False
)

expert_agent = Agent(
    role="Bomb Manual Expert",
    goal="Interpret module descriptions and give precise disarming instructions.",
    backstory=
    """
        You're playing a game where you have to advise a person defusing a bomb.
        USE THE PROVIDED TOOL TO GET MANUAL!!!
        EXAMPLE HOW TO PLAY SIMON MODULE:
        When the sequence is "blue, red" for the first color
        you need to look in round 1 column of table in manual,
        for second colour look in round 2 column of table in manual.
    """,
    tools=[expert_tool],
    llm=USED_MODEL,
    verbose=True,
    cache=False
)

task_defuser_question = Task(
    description=(
        """
            INSTRUCTIONS:
            1. USE defuser_tool with {'command': 'state'} to get bomb state.
            2. Create output for bomb defusal expert.
        """
    ),
    expected_output="""A JSON structure containing bomb data and list of available commands.""",
    agent=defuser_agent
)

task_expert = Task(
    description=(
        """
            INSTRUCTIONS:
            1. Use tool to get manual!
            2. Read bomb description from defuser and compare to manual.
            3. Create output for defuser to take actions.
        """
    ),
    expected_output=
    """ 
        A JSON structure like:
        { 
            actions: [...]
        }
        with only actions from availableActions field from Defuser output in the list. 
    """,
    agent=expert_agent
)

task_expert_shortened = Task(
    description=(
        """
            INSTRUCTIONS:
            1. Read bomb description from defuser and compare to manual.
            2. Create output for defuser to take actions.
        """
    ),
    expected_output=
    """ 
        A JSON structure like:
        { 
            actions: [...]
        }
        with only actions from availableActions field from Defuser output in the list. 
    """,
    agent=expert_agent
)

task_defuser_action = Task(
    description=(
        """
            INSTRUCTIONS:
            1. Perform actions listed in JSON object given by expert using defuser_tool,
            like: {'command': action} for action in actions. PERFORM THIS STEP!!!
            DO THE ACTIONS GIVEN BY EXPERT!
            2. Write bomb state information after performing actions to output.
        """
    ),
    expected_output=
    """
    Describe the bomb state after all actions from expert are performed!   
    """,
    agent=defuser_agent
)

crew = Crew(
    agents=[defuser_agent, expert_agent],
    tasks=[task_defuser_question, task_expert, task_defuser_action],
    verbose=True,
    output_log_file=True,
    cache=False
)

crew_memory = Crew(
    agents=[defuser_agent, expert_agent],
    tasks=[task_defuser_question, task_expert, task_defuser_action,
           task_expert_shortened, task_defuser_action,
           task_expert_shortened, task_defuser_action,
           task_expert_shortened, task_defuser_action,
           task_expert_shortened, task_defuser_action],
    verbose=True,
    output_log_file=True,
    cache=False
)
