from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper


web_search = DuckDuckGoSearchRun()
wikipedia = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)


def search_web(query: str) -> str:
    """Search the web for current information relevant to the user's question."""
    return web_search.run(query)


def search_wikipedia(query: str) -> str:
    """Look up concise background information on Wikipedia."""
    return wikipedia.run(query)
