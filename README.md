# OpenEnv Support Triage

A real-world OpenEnv environment representing a customer support tier 1 triage task.

## Description
The agent embodies a support representative with an inbox of support tickets. The agent must:
- Read incoming tickets.
- Reply to standard support queries.
- Escalate high priority or urgent technical issues to tier 2.
- Mark spam emails to keep the inbox clean.

This tasks models a genuine real-world task performed daily by thousands of customer service agents. 

## Setup
```sh
pip install -r requirements.txt
```

## Running Baseline Inference
```sh
export OPENAI_API_KEY="sk-..."
python inference.py
```

## OpenEnv Validation
```sh
openenv validate
```
