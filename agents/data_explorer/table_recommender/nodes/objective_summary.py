from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.language_models.chat_models import BaseChatModel

from utils.llm import call_llm
from prompts.table_recommender_prompts import objective_summary_prompt as prompt, objective_summary_parser as parser

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env")


def extract_objective_summary(state, llm: BaseChatModel):
    full_text = state.get("parsed_text") or state.get("input")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)
    vectorstore = FAISS.from_texts(chunks, OpenAIEmbeddings())
    top_chunks = vectorstore.similarity_search("What is the purpose of the analysis and what data should be used?", k=5)
    selected = "\n\n".join([doc.page_content for doc in top_chunks])

    try: 
        response = call_llm(
            prompt=prompt,
            parser=parser,
            variables={
                "analysis_text": selected,  # 분석 텍스트
                "format_instructions": parser.get_format_instructions()
            },
            llm=llm
        )
        # print("Objective Summary Response:", response)
        return {
            "objective_summary": f"- Objective: {response.summerized_objective}\n- Data needed: {response.required_data}"
            }
    
    except Exception as e:
        print(f"Error: objective summary LLM call: {e}")
        return {"objective_summary": "Error summarizing objective. Please check the input data."}