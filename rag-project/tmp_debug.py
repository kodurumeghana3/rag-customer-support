from dotenv import load_dotenv
load_dotenv()
import os
print('GROQ_API_KEY', os.getenv('GROQ_API_KEY'))
from src.utils.config import Config
from src.agents.rag_graph import build_graph, run_query
cfg = Config()
config_dict = {
    'embedding_model': cfg.EMBEDDING_MODEL,
    'llm_model': cfg.LLM_MODEL,
    'top_k': cfg.TOP_K,
    'escalation_keywords': cfg.ESCALATION_KEYWORDS,
    'collection_name': cfg.COLLECTION_NAME,
    'chroma_dir': cfg.CHROMA_DIR
}
print('config', config_dict)
graph = build_graph(config_dict)
print('graph built')
result = run_query(graph, 'What is your return policy?', 'test', config_dict)
print('answer', repr(result['answer']))
print('done')
