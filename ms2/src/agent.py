import json
from ms2.src.retrieval import VectorRetriever, GraphRetriever

class ConversationalAgent:
    """ Stateful Conversation Layer for Qwen2.5-3B-Instruct. """
    def __init__(self, model_identifier="Qwen/Qwen2.5-3B-Instruct"):
        self.model_identifier = model_identifier
        self.history = []
        
    def generate_response(self, user_prompt: str, contract_text: str, context: str) -> str:
        """
        Combines external RAG context (from train split) with raw contract text (eval split)
        and conversation history to produce an answer.
        """
        prompt = f"""
System: You are an expert legal aide reviewing NDAs.
Below is the contract currently under review:
[CONTRACT BEGIN]
{contract_text[:1000]}... # Truncated for demonstration
[CONTRACT END]

Below is historical reasoning on similar legal clauses from past precedents (External Context):
[CONTEXT BEGIN]
{context}
[CONTEXT END]

User Question: {user_prompt}
        """
        # Append User Question
        self.history.append({"role": "user", "content": user_prompt})
        
        # Generation Logic utilizing Transformers or vLLM...
        print("Invoking Qwen Model...")
        ai_response = "Here is an analysis based on the extracted precedents..."
        
        self.history.append({"role": "assistant", "content": ai_response})
        return ai_response
    
    def get_history(self) -> list:
        return self.history
