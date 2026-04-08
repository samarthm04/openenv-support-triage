import asyncio
import json
import os
from typing import List

# Inference script must use standard bare OpenAI API Client
from openai import AsyncOpenAI
# openenv_core imports
from env import SupportTriageEnv, TriageAction, TriageObservation

# Mandatory configuration variables mentioned in prompt
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY", "")) # Fallback to openai API key if HF not set

# Project imports
from env import TriageAction, TriageObservation

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    err_str = f" error={error}" if error else ""
    print(f"[STEP] step={step} action={action!r} reward={reward:.2f} done={done}{err_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards}", flush=True)

async def run_task(client: AsyncOpenAI, task_difficulty: str):
    # This matches the validation script's requirement to connect to the environment
    # In a local validate-submission, openenv serve runs on port 7860
    # Wait, the sample script just imports the environment directly?
    # Sample script uses: `env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)` or `from_docker_image`
    # Let me fallback to HTTP EnvClient over 7860.
    
    print(f"[DEBUG] Instantiating SupportTriageEnv")
    
    env = SupportTriageEnv()
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    MAX_STEPS = 20

    log_start(task=f"triage_{task_difficulty}", env="support_triage", model=MODEL_NAME)

    try:
        # Pass task_difficulty via resetkwargs if supported?
        # Actually openenv base client `reset` takes **kwargs. 
        # `EnvClient` from `openenv-core` 0.2.3 has `reset(self, **kwargs)`? 
        # In openenv client interface: `await env.reset(task_difficulty=task_difficulty)`
        # Let's inspect EnvClient signature just in case. 
        
        obs = env.reset(task_difficulty=task_difficulty) 
        
        last_echoed = obs.echoed_message
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            # Prompt to model
            prompt = f"Inbox Status: {obs.inbox_status}\\n"
            if obs.current_ticket_content:
                prompt += f"Current Ticket:\\n{obs.current_ticket_content}\\n"
            prompt += f"Last message: {last_echoed}\\n"
            prompt += "Choose action: {'action_type': 'read'|'reply'|'escalate'|'mark_spam'|'finish', 'ticket_id': <id>, 'reply_text': <text>}\\n"
            prompt += "Reply strictly in JSON."

            messages = [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": prompt}]
            
            completion = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            message_content = completion.choices[0].message.content
            action_dict = json.loads(message_content)
            
            # Action string for logging
            action_str = json.dumps(action_dict)

            # Robust LLM parsing: handle integers and missing prefixes
            if "ticket_id" in action_dict and action_dict["ticket_id"] is not None:
                action_dict["ticket_id"] = str(action_dict["ticket_id"])
                if action_dict["ticket_id"].isdigit():
                    action_dict["ticket_id"] = f"t{action_dict['ticket_id']}"

            action_obj = TriageAction(**action_dict)
            obs = env.step(action_obj)

            reward = obs.reward or 0.0
            done = obs.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_echoed = obs.echoed_message
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = sum(rewards)
        score = min(max(score, 0.0), 1.0)  
        success = score >= 0.99

    finally:
        try:
            env.close()
        except:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main():
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    
    for diff in ["easy", "medium", "hard"]:
        print(f"\\n--- Running Task: {diff} ---")
        await run_task(client, diff)

if __name__ == "__main__":
    asyncio.run(main())
