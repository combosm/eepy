from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from ai.tools import search_web, search_wikipedia
import os

load_dotenv()

# define the response schema using Pydantic BaseModel
class ResearchResponse(BaseModel):
    topic: str
    summary: str


SYSTEM_PROMPT = """You are a helpful AI driving assistant.
Keep responses concise, clear, and suitable for someone who may be driving.
Use tools when current or reference information is needed.
Return the requested structured response."""


def initialise_agent() -> Any:
    """Build a LangChain 1.x agent with validated structured output."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    tools = [search_web, search_wikipedia]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        response_format=ResearchResponse,
    )

ai_response = "AI RESPONSE"


_agent: Any | None = None


def get_agent() -> Any:
    """Return the process-wide agent, building it only on first use."""
    global _agent
    if _agent is None:
        _agent = initialise_agent()
    return _agent


def invoke_agent(
    agent: Any,
    query: str,
    chat_history: list[dict[str, str]],
) -> ResearchResponse:
    """Invoke the LangChain 1.x message-based agent and validate its response."""
    chat_history.append({"role": "user", "content": query})
    result = agent.invoke({"messages": list(chat_history)})
    structured_response = result.get("structured_response")
    if isinstance(structured_response, ResearchResponse):
        response = structured_response
    else:
        response = ResearchResponse.model_validate(structured_response)
    chat_history.append({"role": "assistant", "content": response.summary})
    return response


def answer_once(query: str) -> str:
    """Run one stateless query and return its validated summary."""
    global ai_response
    ai_response = invoke_agent(get_agent(), query, []).summary
    return ai_response


def run_ai(agent: Any) -> None:
    from ai.stt_tts import output_audio, record_audio

    global ai_response
    chat_history: list[dict[str, str]] = []
    
    while True:
        query = record_audio()
        
        if not query:
            continue

        # Record user query and chat history
        # Check if the user wants to exit
        deactivate_words = ["bye", "goodbye", "later"]
        if any(word in query.lower() for word in deactivate_words):
            output_audio("Goodbye!")
            break

        ai_response = invoke_agent(agent, query, chat_history).summary
        if ai_response:
            print(f"User Query: {query}")
            print(f"ai_response: {ai_response}")

            # Convert the ai_response to speech using text-to-speech
            output_audio(ai_response)

def main() -> None:
    from ai.stt_tts import output_audio, record_audio

    agent = initialise_agent()

    # Word for activation
    activate_words = ["hey", "hello", "hi", "yo", "hey jit"]

    while True:    
        print("Listening for activation...")
        detected_text = record_audio()
        print(f"Detected text: {detected_text}")  

        # Check if the wake word is detected
        if detected_text and any(activate_word in detected_text.lower() for activate_word in activate_words):
            output_audio("Hey!")
            # Activate the AI
            run_ai(agent)
        else:
            print("Activate word not detected. Please try again.")

if __name__ == "__main__":
    main()
