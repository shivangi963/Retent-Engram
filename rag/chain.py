"""
rag/chain.py
============
The RAG pipeline written with LangChain (LCEL), as in Figure 4.3 of the report:

    inputs --> retrieve notes --> prompt template --> ChatOllama --> string parser
               (our FAISS store)  (same wording as the    (local LLM)

  * retrieval  : rag.generate.retrieve_chunks -> rag.store (FAISS), restricted to the concept's own notes
  * prompt     : ChatPromptTemplate built from the SAME SYSTEM_PROMPT / HUMAN_TEMPLATE / format text the
                 direct path uses, so both paths ask the model for exactly the same thing
  * LLM        : ChatOllama (langchain-ollama) pointed at the local Ollama server

Enable with RETENT_USE_LANGCHAIN=1 (pip install langchain-core langchain-ollama).  If LangChain is not
installed or the chain fails before producing output, generation falls back to the direct HTTP path.
"""
import os


def enabled() -> bool:
    return os.getenv("RETENT_USE_LANGCHAIN", "0") == "1"


def available() -> bool:
    try:
        import langchain_core                      # noqa: F401
        return True
    except Exception:
        return False


def _default_llm(kind: str):
    from backend.config import OLLAMA_URL
    from rag import generate as gen
    try:
        from langchain_ollama import ChatOllama
    except ImportError:                            # older installs
        from langchain_community.chat_models import ChatOllama
    return ChatOllama(model=gen.pick_model(), base_url=OLLAMA_URL, temperature=0.4,
                      num_predict=gen.TOKEN_CAPS.get(kind, 400), num_ctx=3072, keep_alive="30m")


def build_chain(content_type: str, llm=None):
    """retrieve -> prompt -> llm -> parser.  `llm` can be any LangChain chat model (tests pass a fake one)."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from rag import generate as gen

    def retrieve(inputs: dict) -> str:
        return gen.build_context_string(gen.retrieve_chunks(inputs["concept_id"], inputs["name"]))

    prompt = ChatPromptTemplate.from_messages([("system", gen.SYSTEM_PROMPT), ("human", gen.HUMAN_TEMPLATE)])
    return (RunnablePassthrough.assign(context=RunnableLambda(retrieve))
            | prompt
            | (llm if llm is not None else _default_llm(content_type))
            | StrOutputParser())


def stream_rag(concept_id: str, concept_name: str, content_type: str, difficulty: str, llm=None):
    """Yield the answer token by token."""
    from rag import generate as gen
    inputs = {"concept_id": concept_id, "name": concept_name, "difficulty": difficulty,
              "level": gen.LEVEL[difficulty], "format": gen.format_for(content_type, concept_name)}
    for token in build_chain(content_type, llm).stream(inputs):
        yield token
