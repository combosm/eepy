from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from scripts.ai.tools import search_tool, wiki_tool
import os
import scripts.ai.stt_tts

load_dotenv()

# define the response schema using Pydantic BaseModel
class ResearchResponse(BaseModel):
    topic: str
    summary: str


def initialise_agent():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # initialise the language model
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
   
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)

    # chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant. Keep responses concise.\n{format_instructions}"),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    # define the tools the AI agent can use
    tools = [search_tool, wiki_tool]

    # Create the AI agent with the specified LLM, prompt, and tools
    agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
    return agent, tools


def process_response(raw_response):
    try:
        output_str = raw_response.get("output") 
        # parse the output string into the ResearchResponse model and return the summary
        return ResearchResponse.parse_raw(output_str).summary
    except Exception as e:

        print(f"Error parsing response: {e}")
        print(f"Raw Response: {raw_response}")
        
        # return an error message if parsing fails
        return "Sorry, I couldn't process the response."

ai_response = "AI RESPONSE"


_agent_executor = None


def get_agent_executor():
    """Return the process-wide AgentExecutor, building it on first use."""
    global _agent_executor
    if _agent_executor is None:
        agent, tools = initialise_agent()
        _agent_executor = AgentExecutor(agent=agent, tools=tools)
    return _agent_executor


def invoke_agent(agent_executor, query, chat_history):
    """record one turn in `chat_history` and return the agent's raw response."""
    chat_history.append({"role": "user", "content": query})
    raw_response = agent_executor.invoke({"query": query, "chat_history": chat_history})
    chat_history.append({"role": "assistant", "content": raw_response.get("output")})
    return raw_response


def answer_once(query):
    """run a single query through the agent and return the parsed answer"""
    global ai_response
    ai_response = process_response(invoke_agent(get_agent_executor(), query, []))
    return ai_response


def run_ai(agent, tools):
    global ai_response
    # Create an executor to run the agent with the tools
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    chat_history = []
    
    while True:
        query = scripts.ai.stt_tts.record_audio()
        
        if not query:
            continue

        # Record user query and chat history
        raw_response = invoke_agent(agent_executor, query, chat_history)

        # Check if the user wants to exit
        deactivate_words = ["bye", "goodbye", "later"]
        if any(word in query.lower() for word in deactivate_words):
            scripts.ai.stt_tts.output_audio("Goodbye!")
            break

        # Process the response
        ai_response = process_response(raw_response)
        if ai_response:
            print(f"User Query: {query}")
            print(f"ai_response: {ai_response}")

            # Convert the ai_response to speech using text-to-speech
            scripts.ai.stt_tts.output_audio(ai_response)

def main():
    agent, tools = initialise_agent()

    # Word for activation
    activate_words = ["hey", "hello", "hi", "yo", "hey jit"]

    while True:    
        print("Listening for activation...")
        detected_text = scripts.ai.stt_tts.record_audio()
        print(f"Detected text: {detected_text}")  

        # Check if the wake word is detected
        if detected_text and any(activate_word in detected_text.lower() for activate_word in activate_words):
            scripts.ai.stt_tts.output_audio("Hey!")
            # Activate the AI
            run_ai(agent, tools)  
        else:
            print("Activate word not detected. Please try again.")

if __name__ == "__main__":
    main()