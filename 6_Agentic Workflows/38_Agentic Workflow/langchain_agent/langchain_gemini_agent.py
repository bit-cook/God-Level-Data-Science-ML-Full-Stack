import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 1. define tools
@tool
def search_web(query: str) -> str:
    """Searches the web using DuckDuckGo for the given query"""
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
        return str(results)

@tool
def get_stock_price(ticker: str) -> str:
    """Gets the latest stock price of a given ticker symbol"""
    import yfinance as yf
    stock = yf.Ticker(ticker)
    price = stock.history(period="1d")['Close'].iloc[-1]
    return f"The latest price of {ticker} is ${price:.2f}"

tools = [search_web, get_stock_price]

# 2. create a planner (LLM + prompt)
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0)

template = """
Answer the following questions as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

# 3. create the agent logic
agent = create_react_agent(llm, tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. run the agent
if __name__ == "__main__":
    goal = "What is the current stock price of NVIDIA(NVDA) and what are the latest headlines about the company?"
    result = agent_executor.invoke({"input": goal})
    print("----Final Result----")
    print(result['output'])
