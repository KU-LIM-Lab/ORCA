import os
import asyncio
from typing import Union, Any, overload
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_core.prompts import BasePromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.language_models.chat_models import BaseChatModel


@overload
def call_llm(prompt: str, parser: None = None, variables: dict = None, model: str = "gpt-4o-mini", temperature: float = 0.3) -> str: ...

@overload
def call_llm(prompt: BasePromptTemplate, parser: None = None, variables: dict = None, model: str = "gpt-4o-mini", temperature: float = 0.3) -> str: ...

@overload
def call_llm(prompt: BasePromptTemplate, parser: BaseOutputParser, variables: dict, model: str = "gpt-4o-mini", temperature: float = 0.3) -> Any: ...


def call_llm(
    prompt: Union[str, BasePromptTemplate],
    parser: BaseOutputParser = None,
    variables: dict = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    provider: str = "openai",
    llm: BaseChatModel = None
) -> Union[str, Any]:
    """
    If `llm` is provided, use it directly. Otherwise, create using model/provider info.
    
    General-purpose LLM caller supporting:
    - Plain string prompt → str output
    - PromptTemplate → str output
    - PromptTemplate + Parser → structured output (e.g., BaseModel, dict, list)
    Supports OpenAI and Ollama providers.
    """
    if llm is None:
        llm = get_llm(model=model, temperature=temperature, provider=provider)

    # Case 1: PromptTemplate with parser
    if isinstance(prompt, BasePromptTemplate) and parser:
        if not variables:
            raise ValueError("PromptTemplate with parser requires input variables.")
        chain = prompt | llm | parser
        return chain.invoke(variables)

    # Case 2: PromptTemplate without parser
    elif isinstance(prompt, BasePromptTemplate):
        if not variables:
            raise ValueError("PromptTemplate requires input variables.")
        chain = prompt | llm
        return chain.invoke(variables).content.strip()

    # Case 3: Plain string prompt
    elif isinstance(prompt, str):
        return llm.invoke(prompt).content.strip()

    raise TypeError("Prompt must be a string or a BasePromptTemplate.")
    

# Async versions of the LLM functions
async def call_llm_async(
    prompt: Union[str, BasePromptTemplate],
    parser: BaseOutputParser = None,
    variables: dict = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    provider: str = "openai",
    llm: BaseChatModel = None
) -> Union[str, Any]:

    if llm is None:
        llm = get_llm(model=model, temperature=temperature, provider=provider)

    # Case 1: PromptTemplate with parser
    if isinstance(prompt, BasePromptTemplate) and parser:
        if not variables:
            raise ValueError("PromptTemplate with parser requires input variables.")
        chain = prompt | llm | parser
        return await chain.ainvoke(variables)

    # Case 2: PromptTemplate without parser
    elif isinstance(prompt, BasePromptTemplate):
        if not variables:
            raise ValueError("PromptTemplate requires input variables.")
        chain = prompt | llm
        result = await chain.ainvoke(variables)
        return result.content.strip()

    # Case 3: Plain string prompt
    elif isinstance(prompt, str):
        result = await llm.ainvoke(prompt)
        return result.content.strip()

    raise TypeError("Prompt must be a string or a BasePromptTemplate.")


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.3, provider: str = "openai") -> BaseChatModel:
    """
    Returns an LLM instance based on the specified provider.
    Supports OpenAI and Ollama.
    """
    if provider == "openai":
        return ChatOpenAI(model_name=model, temperature=temperature)
    
    elif provider == "ollama":
        return OllamaLLM(
            model=model, 
            temperature=temperature,
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434")
)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'ollama'.")