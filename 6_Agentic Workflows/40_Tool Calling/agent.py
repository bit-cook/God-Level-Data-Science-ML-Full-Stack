import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, Tool, create_react_agent

import requests

print("setiing up env and loading api key")
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("No GOOGLE_API_KEY found in environment variables")

print("API key loaded")

llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash", api_key=google_api_key)
print("LLM initialized")

# get current weather using any free weather API
# add a caculator tool

# get exchange rates

def get_exchange_rate(query: str) -> str:
    print(f"Fetching exchange rate for: {query}")
    try:
        parts = query.strip().split()
        if len(parts) != 2:
            return "Please provide input in the format: '<base_currency> <target_currency>'. Example: 'USD EUR'"

        base_currency, target_currency = parts
        url = f"https://api.exchangerate-api.com/v4/latest/{base_currency.upper()}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if target_currency.upper() in data['rates']:
            rate = data['rates'][target_currency.upper()]
            return f"The exchange rate from {base_currency.upper()} to {target_currency.upper()} is {rate}."
        else:
            return f"Could not find exchange rate for {target_currency.upper()}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching exchange rate: {e}"


exchange_rate_tool = Tool(
    name="ExchangeRateConverter",
    func=get_exchange_rate,
    description="Useful for getting the exchange rate. Input format: '<base_currency> <target_currency>' e.g. 'USD EUR'"
)

tools = [exchange_rate_tool]

print("Tool is defined")

prompt_template = PromptTemplate.from_template(
    """You are an AI assistant specialized in answering questions. You have access to the following tools:
    {tools}

    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """
)

agent = create_react_agent(
    llm,
    tools,
    prompt_template
)

agent_executor = AgentExecutor(
    agent = agent,
    tools = tools,
    verbose = True,
    handle_parsing_errors = True
)
print("Agent executor created")


print("running the react agent")

query_exchange = "What is the exchange rate between USD and EUR?"
print(f"Query: {query_exchange}")
result_exchange = agent_executor.invoke({"input": query_exchange})
print(f"Result: {result_exchange['output']}")


query_exchange = "What is the exchange rate between GBP and JPY?"
print(f"Query: {query_exchange}")
result_exchange = agent_executor.invoke({"input": query_exchange})
print(f"Result: {result_exchange['output']}")
