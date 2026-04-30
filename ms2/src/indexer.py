import json
import os
import networkx as nx
from typing import List, Dict, Any

class CorpusIndexer:
    """
    Indexes the ContractNLI training split into a Vector and Graph knowledge base.
    Extracts hypothesis, outcome, and the specific evidence span instead of full raw documents.
    """
    def __init__(self, train_data_path: str):
        self.train_data_path = train_data_path
        self.documents = []
        self.labels_map = {}
        self._load_data()
        
    def _load_data(self):
        with open(self.train_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.documents = data.get('documents', [])
            self.labels_map = data.get('labels', {})
            
    def _extract_evidence_text(self, doc: Dict, span_indices: List) -> List[str]:
        text = doc['text']
        doc_spans = doc.get('spans', [])
        evidence_texts = []
        for idx in span_indices:
            if idx < len(doc_spans):
                span = doc_spans[idx]
                if isinstance(span, list) and len(span) == 2:
                    start, end = span
                    evidence_texts.append(text[start:end])
                elif isinstance(span, str):
                    evidence_texts.append(span)
        return evidence_texts

    def extract_structured_chunks(self) -> List[Dict[str, Any]]:
        """
        Produce a list of structured knowledge chunks for vector RAG.
        Each chunk represents a historical legal evaluation of a specific clause against a hypothesis.
        """
        chunks = []
        for doc in self.documents:
            annotations = doc.get('annotation_sets', [{}])[0].get('annotations', {})
            for hyp_id, ann in annotations.items():
                choice = ann.get('choice')
                span_indices = ann.get('spans', [])
                
                evidence_texts = self._extract_evidence_text(doc, span_indices)
                hypothesis_info = self.labels_map.get(hyp_id, {})
                hypothesis_text = hypothesis_info.get('hypothesis', '')
                
                # Only keep chunks that actually have evidence and mention context
                if choice in ['Entailment', 'Contradiction'] and evidence_texts:
                    for ev_text in evidence_texts:
                        chunk_text = (
                            f"Hypothesis: '{hypothesis_text}'\n"
                            f"Historical Outcome: {choice}\n"
                            f"Supporting Clause: '{ev_text}'"
                        )
                        chunks.append({
                            "doc_id": doc.get('id'),
                            "hyp_id": hyp_id,
                            "choice": choice,
                            "hypothesis": hypothesis_text,
                            "evidence": ev_text,
                            "chunk_text": chunk_text
                        })
        return chunks

    def build_vector_index(self):
        """
        TODO: Initialize Chroma/FAISS and embed the chunks using Qwen or sentence-transformers.
        """
        chunks = self.extract_structured_chunks()
        print(f"Extracted {len(chunks)} evidence chunks for Vector Indexing.")
        # Embedding logic here...

    def build_graph_index(self):
        """
        TODO: Build a knowledge graph.
        Nodes: Hypothesis rules, Found Clauses, Outcomes.
        Edges: relationships such as [Clause] --ENTAILS--> [Hypothesis].
        """
        chunks = self.extract_structured_chunks()
        G = nx.DiGraph()
        
        for chunk in chunks:
            clause_node = f"Clause({chunk['doc_id']}): {chunk['evidence'][:30]}..."
            hyp_node = f"Rule: {chunk['hypothesis']}"
            
            G.add_node(clause_node, type="Clause", text=chunk['evidence'])
            G.add_node(hyp_node, type="Hypothesis")
            G.add_edge(clause_node, hyp_node, relation=chunk['choice'])
            
        print(f"Built Knowledge Graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

if __name__ == "__main__":
    # Test on sample.json to ensure logic works before running on full train.json
    indexer = CorpusIndexer("sample.json")
    indexer.build_vector_index()
    indexer.build_graph_index()
