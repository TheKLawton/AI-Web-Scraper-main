# parse.py
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

template = (
    "You are given page text: {dom_content}\n"
    "Extract ONLY the data that matches: {parse_description}\n"
    "Output MUST be a pure JSON array of objects (no markdown, no code fences, no prose).\n"
    "Each object should use these keys when available: "
    "name, price, image_url, url, category, brand, sku.\n"
    "If a field is missing, use an empty string.\n"
)

model = Ollama(model="llama3")

def parse_with_ollama(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    parts = []
    for i, chunk in enumerate(dom_chunks, start=1):
        resp = chain.invoke({"dom_content": chunk, "parse_description": parse_description})
        print(f"Parsed batch: {i} of {len(dom_chunks)}")
        text = getattr(resp, "content", resp)
        parts.append(text if isinstance(text, str) else str(text))

    # Join all chunk results; we'll parse/merge in main.py
    return "\n".join(parts)
