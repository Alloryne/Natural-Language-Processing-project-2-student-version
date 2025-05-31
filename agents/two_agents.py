import asyncio
import pickle
from typing import Dict
from tqdm import tqdm
import torch

from agents.prompts import expert_prompt, defuser_prompt
from game_mcp.game_client import Defuser, Expert, Resetter
from agents.models import HFModel, SmollLLM

USED_MODEL = "Qwen/Qwen3-0.6B"


async def run_two_agents(
        defuser_model: HFModel,
        expert_model: HFModel,
        server_url: str = "http://0.0.0.0:8080",
        max_new_tokens: int = 50,
        iteration_limit: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        mode: str = 'default',
        quiet: bool = False
) -> Dict[str, int]:
    """
    Main coroutine that orchestrates two LLM agents (Defuser and Expert)
    interacting with the bomb-defusal server.

    :param defuser_model: The HFModel for the Defuser's role.
    :param expert_model: The HFModel for the Expert's role.
    :param server_url: The URL where the bomb-defusal server is running.
    :param max_new_tokens: Max tokens to generate for each LLM response.
    :param iteration_limit: A limit on how many iterations the models will talk for
    (for preventing endless loops)
    :param temperature: both models' temperature.
    :param top_p: both models' top_p.
    :param top_k: both models' top_k.
    :param mode: How model prompt will be structured.
    :param quiet: How much debug info function writes.
    """
    defuser_client = Defuser()
    expert_client = Expert()
    resetter_client = Resetter()
    await resetter_client.connect_to_server(server_url)
    await resetter_client.run('wire')

    iteration_count = 0
    success = -1

    try:
        # 1) Connect both clients to the same server
        await defuser_client.connect_to_server(server_url)
        await expert_client.connect_to_server(server_url)

        while iteration_count < iteration_limit:
            # 2) Defuser checks the bomb's current state
            bomb_state = await defuser_client.run("state")
            if not quiet:
                print("[DEFUSER sees BOMB STATE]:")
                print(bomb_state)

            if "Bomb disarmed!" in bomb_state or "Bomb exploded!" in bomb_state:
                break

            # 3) Defuser formulates question
            def_messages = defuser_prompt(bomb_state, '', mode, 0)
            def_question = defuser_model.generate_response(
                def_messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True
            )

            if not quiet:
                print("[DEFUSER SAYS TO EXPERT]:")
                print(def_question)

            # 4) Expert retrieves the relevant manual text
            manual_text = await expert_client.run()
            if not quiet:
                print("[EXPERT sees MANUAL]:")
                print(manual_text)

            # 5) Expert LLM uses the manual text + defuser’s question
            #    to generate instructions
            exp_messages = expert_prompt(manual_text, def_question, mode)
            expert_advice = expert_model.generate_response(
                exp_messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True
            )
            if not quiet:
                print("\n[EXPERT ADVICE to DEFUSER]:")
                print(expert_advice)

            # 6) Defuser LLM uses the bomb state + expert advice to pick a single action
            def_messages = defuser_prompt(bomb_state, expert_advice, mode, 1)
            def_action_raw = defuser_model.generate_response(
                def_messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True
            )

            # 7) Attempt to extract a known command from def_action_raw
            #    If no recognized command is found, default to "help"
            action = "help"
            for line in def_action_raw.splitlines():
                line = line.strip().lower()
                if line.startswith(("cut", "press", "hold", "release", "help", "state")):
                    action = line.strip()
                    break

            if not quiet:
                print("\n[DEFUSER ACTION DECIDED]:", action)

            # 7) Send that action to the server
            result = await defuser_client.run(action)

            if not quiet:
                print("[SERVER RESPONSE]:")
                print(result)
                print("-" * 60)

            iteration_count += 1

            if "BOMB SUCCESSFULLY DISARMED" in result:
                success = 1
                break
            elif "BOMB HAS EXPLODED" in result:
                success = 0
                break
            elif "Unknown command" in result:
                break

    finally:
        if not quiet:
            print(iteration_count)
        await defuser_client.cleanup()
        await expert_client.cleanup()
        await resetter_client.cleanup()
        return {
            'iterations': iteration_count,
            'success': success,
        }


def default_main():
    defuser_checkpoint = USED_MODEL
    expert_checkpoint = USED_MODEL

    defuser_model = SmollLLM(defuser_checkpoint, device="cpu")
    expert_model = SmollLLM(expert_checkpoint, device="cpu")

    asyncio.run(
        run_two_agents(
            defuser_model=defuser_model,
            expert_model=expert_model,
            server_url="http://127.0.0.1:8080",
            max_new_tokens=500,
            mode='natural'
        )
    )


# Function for performing task 2
def full_eval_main():
    defuser_checkpoint = USED_MODEL
    expert_checkpoint = USED_MODEL

    torch.cuda.empty_cache()
    defuser_model = SmollLLM(defuser_checkpoint, device="cpu")
    expert_model = SmollLLM(expert_checkpoint, device="cpu")

    results = {}

    modes = ['natural', 'markdown', 'json']
    temperatures = [0.2, 0.7, 1.]
    top_ps = [0.3, 0.6, 0.9]
    top_ks = [25, 50, 75]

    ATTEMPTS_NUM = 1

    for mode in modes:
        for temperature in temperatures:
            for top_p in top_ps:
                for top_k in top_ks:
                    key = (mode, temperature, top_p, top_k)
                    iterations_list = []
                    success_list = []

                    for _ in tqdm(range(ATTEMPTS_NUM), desc=f"Mode={mode}, T={temperature}, p={top_p}, k={top_k}"):
                        result = asyncio.run(
                            run_two_agents(
                                defuser_model=defuser_model,
                                expert_model=expert_model,
                                server_url="http://127.0.0.1:8080",
                                max_new_tokens=50,
                                mode=mode,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                quiet=True,
                                iteration_limit=3
                            )
                        )
                        iterations_list.append(result['iterations'])
                        success_list.append(result['success'])

                    results[key] = {
                        'iterations': iterations_list,
                        'success': success_list
                    }

    with open("../results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    full_eval_main()
