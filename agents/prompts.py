from typing import List, Dict


def defuser_prompt(bomb_state: str, expert_advice: str, mode: str, stage: int) -> List[Dict[str, str]]:
    """
    Build a 'messages' list for the Defuser LLM.

    :param bomb_state: Current bomb state text from the server.
    :param expert_advice: Instructions from the Expert.
    :param mode: Decides what prompt structure will be.
    :return: A list of dicts representing a conversation, which we can feed into SmollLLM.generate_response().
    """

    if stage == 0:
        if mode == 'natural':
            system_msg = (
                """
                You are the bomb defuser.
                You get a description of the bomb.
                Describe the bomb to expert and ask which action you should take in regards to the bomb.
                DO NOT WRITE ANYTHING ELSE.
                """
            )
        elif mode == 'markdown':
            system_msg = (
                """
                # PROBLEM STATEMENT
                You are seeing a description of a bomb.
                ## INSTRUCTIONS
                Describe the bomb to an expert in this format, so they can tell you how to defuse it.
                """
            )
        elif mode == 'json':
            system_msg = (
                """
                {
                    "role": "System",
                    "General information": 
                        "
                        You are seeing a description of a bomb.
                        ",
                    "Instructions":
                        "
                        Create a JSON object describing the bomb for expert.
                        "
                }
                """
            )
        else:
            system_msg = (
                "You are the responsible and not harmful assistant."
            )
    else:
        if mode == 'natural':
            system_msg = (
                """
                Use the actions that expert told you to use to defuse in order.
                """
            )
        elif mode == 'markdown':
            system_msg = (
                """
                ## INSTRUCTIONS
                Use the actions that expert told you to use to defuse in order.
                """
            )
        elif mode == 'json':
            system_msg = (
                """
                {
                    "role": "System",
                    "Instructions":
                        "
                        Execute all the commands given by expert in order. Use proper formatting. 
                        "
                }
                """
            )
        else:
            system_msg = (
                "You are the responsible and not harmful assistant."
            )

    user_content = (
        f"Current bomb state:\n{bomb_state}\n\n"
        f"Expert's advice:\n{expert_advice}\n\n"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]
    return messages


def expert_prompt(manual_text: str, defuser_question: str, mode: str) -> List[Dict[str, str]]:
    """
    Build a 'messages' list for the Expert LLM.

    :param manual_text: The text from the bomb manual (server).
    :param defuser_question: A description of what the Defuser sees or asks.
    :param mode: Decides what prompt structure will be.
    :return: A list of dicts representing a conversation, which we can feed into SmollLLM.generate_response().
    """
    if mode == 'natural':
        system_msg = (
            """
                You are playing a game where you have to advise another person how to defuse a bomb 
                based on manual and descripton of the bomb from the other person.
                Consider the information given carefully and give good advice!
                Try to highlight and repeat relevant information!
                Try not to confuse the other person as to what are the valid commands!
                If the the module is stated to be simon module, then the description contains the following error!!!
                The button to be clicked does NOT depend on the round number,
                but on the number of the colour shown in order!
                BE VERY CONCISE!!!!
                GIVE SHORT STATEMENTS!!!!!
            """
        )
    elif mode == 'markdown':
        system_msg = (
            """
            # PROBLEM STATEMENT
                You are playing a game where you have to advise another person how to defuse a bomb 
                based on manual and descripton of the bomb from the other person.
                ## OBSERVATIONS
                The availible actions are always listed after "Availible actions:" line.
                The state of the bomb the other person has to defuse
                is after line containing string BOMB STATE.
                If the the module is stated to be simon module, then the description contains the following error!!!
                The button to be clicked does NOT depend on the round number,
                but on the number of the colour shown in order!
                This is highly important!
                ## INSTRUCTIONS
                Read and consider the information you're given carefully!
                Try to highlight and repeat relevant information!
                Try not to confuse the other person as to what are the valid commands.
                BE VERY CONCISE!!!!
                GIVE SHORT STATEMENTS!!!!!
            """
        )
    elif mode == 'json':
        system_msg = (
            """
            {
                "role": "System",
                "General information": 
                    "
                    You are playing a game where you have to advise another person how to defuse a bomb 
                    based on manual and descripton of the bomb from the other person.
                    ",
                "Advice": 
                    "
                    The availible actions are always listed after "Availible actions:" line.
                    The state of the bomb you have to defuse is after line containing string BOMB STATE.
                    If the the module is stated to be simon module, then the description contains the following error!!!
                    The button to be clicked does NOT depend on the round number,
                    but on the number of the colour shown in order!
                    This is highly important!
                    ",
                "Instructions":
                    "
                    Create JSON object with list of actions for defuser to take,
                    "
            }
            """
        )
    else:
        system_msg = (
            "You are the responsible and not harmful assistant."
        )

    user_content = (
        f"Manual excerpt:\n{manual_text}\n\n"
        f"DEFUSER sees or asks:\n{defuser_question}\n\n"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]
    return messages
