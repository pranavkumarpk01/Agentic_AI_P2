"""
api.py  -  the API half: get LIVE information from a third-party service.

We use Tavily, a web-search API built for LLMs.
It is just an HTTP POST that returns JSON - nothing magical.

Test it alone:   python api.py "latest changes to maternity benefit law India"
"""
import sys
import requests
from config import TAVILY_API_KEY


def search_web(query: str, max_results: int = 3) -> list[dict]:
    """Return [{"title":..., "url":..., "content":...}, ...] from the live web."""
    if not TAVILY_API_KEY:
        raise SystemExit("TAVILY_API_KEY is missing. Add it to your .env file.")

    resp = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},   # authentication
        json={"query": query, "max_results": max_results},       # request body
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()["results"]
    return [{"title": r["title"], "url": r["url"], "content": r["content"]} for r in results]


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "latest changes to maternity benefit law India"
    print(f"API query: {query}\n")
    for i, r in enumerate(search_web(query), 1):
        print(f"#{i}  {r['title']}\n    {r['url']}\n    {r['content'][:200]}...\n")
