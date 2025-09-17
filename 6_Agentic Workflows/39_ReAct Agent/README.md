## Problem - Dumb LLM

Prompt
"If a pizza costs $18.75 and I want to buy 3, plus a 15% tip, what is the total cost?"

limitation of llms
- they dont have calculators
- they dont have databases
- dont have web browsers

## ReAct Framework

Loop:

Reason(Thought) -> Act(Action) -> Observe(Observation)

Hands-On Steps to create a ReAct Agent which use calculator tool for precise answers

1. installing and env, and loading api keys
2. degining a tool
3. create a planner (llm - brain)
4. the prompt - agent's instructions
5. creating the agent
6. run the agent



Assignment

- create a agent with 3 tools
- web search
- calculator
- access an api