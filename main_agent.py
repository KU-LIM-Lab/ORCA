from pathlib import Path
import sys, os, yaml, argparse, datetime, traceback
# Fix OpenMP duplicate library error on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableMap

from utils.llm import get_llm
from utils.load_causal_graph import load_causal_graph
from utils.prettify import print_final_output_explorer, print_final_output_recommender, print_final_output_sql, print_final_output_causal

from agents.data_explorer.table_explorer import generate_description_graph
from agents.data_explorer.table_recommender import generate_table_recommendation_graph
from agents.data_explorer.text2sql_generator import generate_text2sql_graph 
from agents.causal_analysis import generate_causal_analysis_graph
from experiments.causal_analysis.causal_pre_information import DEFAULT_EXPRESSION_DICT


# setup env
sys.path.append(os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Tee:
    def __init__(self, *streams): 
        self.streams = streams
    def write(self, data): 
        [s.write(data) or s.flush() for s in self.streams]
    def flush(self): 
        [s.flush() for s in self.streams]

def read_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f: 
            return f.read()
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

class Agent:
    def __init__(self, config, prompt_routing_path, db_id=None):
        self.config = config
        self.routing_template = read_file(prompt_routing_path)
        self.followup_count = 0
        self.followup_max = 2  # 최대 2번만 followup 허용

        # db 설정
        if db_id:
            self.db_id = db_id
        else:
            db_config = config.database
            self.db_type = db_config.type.lower()

            if self.db_type == "postgresql":
                self.db_id = db_config.postgresql.dbname

            elif self.db_type == "sqlite":
                sqlite_path = db_config.sqlite.sqlite_path
                if "{db_name}" in sqlite_path:
                    self.db_id = "daa"  
                else:
                    self.db_id = Path(sqlite_path).stem  # e.g. "data/daa.sqlite" → "daa"
            else:
                self.db_id = "daa"
        

        self.llm = get_llm(
            model = "gpt-4o-mini", # config.llm.model
            temperature=0.7, 
            provider= "openai") # config.llm.provider) 
        self.memory = ConversationBufferMemory(
             return_messages=True
        )
        
        # Initialize the routing chain
        self.routing_chain = (
            RunnableMap({
                "history": RunnableLambda(lambda x: self.memory.load_memory_variables({})["history"]),
                "user_input": lambda x: x["user_input"]
            })
            | ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{user_input}")
            ])
            | self.llm
        )
        
    def route_task(self, user_input: str, add_to_memory=True) -> tuple[str, str]:
        # accept file input
        if user_input.strip().endswith((".pdf", ".docx", ".pptx")):
            return "recommend", user_input.strip()  # 바로 recommend로 넘김
        
        if not self.routing_template:
            print("⚠️ No routing prompt template found.")
            return "clarify", user_input

        # 사용자 입력 메시지 저장 (memory 반영)
        if add_to_memory:
            self.memory.chat_memory.add_user_message(user_input)

        try:
            reply_msg = self.routing_chain.invoke({
                "user_input": self.routing_template.format(user_input=user_input)
                })
            
            # LLM 응답이 str인지 LangChain 메시지 객체인지에 따라 처리
            if isinstance(reply_msg, str):
                reply = reply_msg.strip()
            else:
                reply = reply_msg.content.strip()

            self.memory.chat_memory.add_ai_message(reply)
            
        except Exception as e:
            print(f"[LangChain Routing Error] {e}")
            return "clarify", user_input

        # print(f"[DEBUG] Routing response: {reply}")

        if reply.startswith("describe:"):
            return "describe", reply.split("describe:")[1].strip()
        elif reply == "recommend":
            return "recommend", user_input
        elif reply == "text2sql":
            return "text2sql", user_input
        elif reply == "causal_analysis":
            return "causal_analysis", user_input
        elif reply.startswith("followup:"):
            self.followup_count += 1
            if self.followup_count > self.followup_max:
                print("❗️ Too many follow-up questions. Please rephrase your original question.")
                return "clarify", user_input # fallback
            
            question = reply.split("followup:")[1].strip()
            print(f"🤖 Just to clarify: {question}")
            user_followup = input("🧑 Your clarification: ")
                                  
            self.memory.chat_memory.add_user_message(user_followup)    
            return self.route_task(user_followup, add_to_memory = False) # recursive
        else:
            return "clarify", user_input

    def run_task(self, task: str, content: str):
        try:
            if task == "describe":
                app = generate_description_graph(llm = self.llm)
                result = app.invoke({"input": content, "db_id": self.db_id})
                final_output = print_final_output_explorer(result["final_output"])
                
                self.memory.chat_memory.add_ai_message(final_output)
                
                return final_output
            
            elif task == "recommend":
                app = generate_table_recommendation_graph(llm = self.llm)
                result = app.invoke({"input": content, "db_id": self.db_id})
                final_output = print_final_output_recommender(result["final_output"])
                
                self.memory.chat_memory.add_ai_message(final_output)
                
                return final_output
            
            elif task == "text2sql": # Placeholder for task 4
                app = generate_text2sql_graph(llm = self.llm)

                initial_state = {
                    "query": content,
                    "messages": [],
                    "desc_str": None,
                    "fk_str": None,
                    "extracted_schema": None,
                    "final_sql": None,
                    "qa_pairs": None,
                    "pred": None,
                    "result": None,
                    "error": None,
                    "pruned": False,
                    "send_to": "selector_node",
                    "try_times": 0,
                    "llm_review": None,
                    "review_count": 0,
                    "output": None,
                    "db_id": self.db_id 
                }

                result = app.invoke(initial_state)
                final_output = print_final_output_sql(result["output"])

                return final_output
            
            elif task == "causal_analysis":
                CAUSAL_GRAPH_PATH = "experiments/causal_analysis/causal_graph_full.json"
                
                # 그래프 및 변수 매핑 로드
                causal_graph = load_causal_graph(CAUSAL_GRAPH_PATH)
                expression_dict = DEFAULT_EXPRESSION_DICT

                app = generate_causal_analysis_graph(
                    llm=self.llm)
                
                result = app.invoke({
                    "input": content,
                    "db_id": self.db_id,
                    "causal_graph": causal_graph,
                    "expression_dict": expression_dict
                })
                
                final_output = print_final_output_causal(result)
                self.memory.chat_memory.add_ai_message(final_output)

                return final_output
            
            else:
                fallback_msg = "🤖 I'm not sure what task to run. Could you clarify your request?"
                self.memory.chat_memory.add_ai_message(fallback_msg)
                return fallback_msg
            
        except Exception:
            error_msg =  "❌ An error occurred during task execution:\n" + traceback.format_exc()
            self.memory.chat_memory.add_ai_message(error_msg)
            return error_msg

    def execute(self, user_input: str) -> str:
        self.followup_count = 0  # Reset follow-up count for each new input
        task, content = self.route_task(user_input)
        
        # print("\n [DEBUG] Memory state (conversation history):")
        # for m in self.memory.chat_memory.messages:
        #     prefix = "👤" if isinstance(m, HumanMessage) else "🤖"
        #     print(f"{prefix} {m.content}")
        
        result = self.run_task(task, content)
        return result

def dict2namespace(config):
    ns = argparse.Namespace()
    for k, v in config.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns

def run_agent_loop(config, input_dir, db_id=None):
    routing_prompt_path = os.path.join(input_dir, config.routing_prompt + ".txt")
    if not os.path.exists(routing_prompt_path):
        raise FileNotFoundError(f"Routing prompt not found: {routing_prompt_path}")
    
    agent = Agent(config, routing_prompt_path, db_id=db_id)
    
    print("🤖 Hi! Please enter your question or upload a file path (e.g., './analysis_plan.pdf'). Type 'exit' to quit.")
    while True:
        user_input = input("🧑 Input: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        
        try:
            result = agent.execute(user_input)
            print(result)

            # 로그 기록
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("logs/session.log", "a", encoding="utf-8") as f:
                f.write(f"\n[{timestamp}] 👤 {user_input}\n")
                f.write(f"[{timestamp}] 🤖 {result}\n")

        except Exception as e:
            print(f"❌ An error occurred: {e}")
            traceback.print_exc()
    

if __name__ == "__main__":
    CONFIG_PATH = "_config.yml"
    INPUT_DIR = "prompts/"
    
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = Tee(sys.__stdout__, open(log_file, "a", encoding="utf-8"))

    with open(CONFIG_PATH, "r") as f:
        config = dict2namespace(yaml.safe_load(f))

    run_agent_loop(config, INPUT_DIR)