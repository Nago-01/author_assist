"""
Curated domain dictionary for high-precision tag lookup.
Edit this file to extend coverage — no code changes needed elsewhere.
"""

from __future__ import annotations

# Structure:    
GAZETTEER: dict[str, tuple[str, str]] = {
    # AI/ML Concepts
    "transformer": ("Transformer", "AI/ML Concept"),
    "transformers": ("Transformer", "AI/ML Concept"),
    "attention mechanism": ("Attention Mechanism", "AI/ML Concept"),
    "self-attention": ("Self-Attention", "AI/ML Concept"),
    "large language model": ("Large Language Model", "AI/ML Concept"),
    "large language models": ("Large Language Model", "AI/ML Concept"),
    "llm": ("Large Language Model", "AI/ML Concept"),
    "llms": ("Large Language Model", "AI/ML Concept"),
    "retrieval-augmented generation": ("Retrieval-Augmented Generation", "AI/ML Concept"),
    "rag": ("Retrieval-Augmented Generation", "AI/ML Concept"),
    "fine-tuning": ("Fine-Tuning", "AI/ML Concept"),
    "finetuning": ("Fine-Tuning", "AI/ML Concept"),
    "lora": ("LoRA", "AI/ML Concept"),
    "rlhf": ("RLHF", "AI/ML Concept"),
    "reinforcement learning from human feedback": ("RLHF", "AI/ML Concept"),
    "mixture of experts": ("Mixture of Experts", "AI/ML Concept"),
    "moe": ("Mixture of Experts", "AI/ML Concept"),
    "prompt engineering": ("Prompt Engineering", "AI/ML Concept"),
    "chain-of-thought": ("Chain-of-Thought", "AI/ML Concept"),
    "chain of thought": ("Chain-of-Thought", "AI/ML Concept"),
    "in-context learning": ("In-Context Learning", "AI/ML Concept"),
    "few-shot learning": ("Few-Shot Learning", "AI/ML Concept"),
    "zero-shot": ("Zero-Shot Learning", "AI/ML Concept"),
    "embedding": ("Embeddings", "AI/ML Concept"),
    "embeddings": ("Embeddings", "AI/ML Concept"),
    "vector database": ("Vector Database", "AI/ML Concept"),
    "neural network": ("Neural Network", "AI/ML Concept"),
    "deep learning": ("Deep Learning", "AI/ML Concept"),
    "machine learning": ("Machine Learning", "AI/ML Concept"),
    "natural language processing": ("NLP", "AI/ML Concept"),
    "nlp": ("NLP", "AI/ML Concept"),
    "computer vision": ("Computer Vision", "AI/ML Concept"),
    "diffusion model": ("Diffusion Model", "AI/ML Concept"),
    "generative ai": ("Generative AI", "AI/ML Concept"),
    "multimodal": ("Multimodal AI", "AI/ML Concept"),
    "benchmark": ("Benchmarking", "AI/ML Concept"),
    "benchmarking": ("Benchmarking", "AI/ML Concept"),
    "evaluation": ("Model Evaluation", "AI/ML Concept"),
    "combinatorial optimization": ("Combinatorial Optimization", "AI/ML Concept"),
    "graph neural network": ("Graph Neural Network", "AI/ML Concept"),
    "gnn": ("Graph Neural Network", "AI/ML Concept"),
    "knowledge graph": ("Knowledge Graph", "AI/ML Concept"),
    "agent": ("AI Agent", "AI/ML Concept"),
    "agentic": ("Agentic AI", "AI/ML Concept"),
    "multi-agent": ("Multi-Agent System", "AI/ML Concept"),
    "tool use": ("Tool Use", "AI/ML Concept"),
    "function calling": ("Function Calling", "AI/ML Concept"),

    # Researchers
    "hinton": ("Geoffrey Hinton", "Researcher"),
    "lecun": ("Yann LeCun", "Researcher"),
    "bengio": ("Yoshua Bengio", "Researcher"),
    "karpathy": ("Andrej Karpathy", "Researcher"),
    "vaswani": ("Ashish Vaswani", "Researcher"),
    "brown": ("Tom Brown", "Researcher"),

    # Organizations
    "openai": ("OpenAI", "Organization"),
    "deepmind": ("DeepMind", "Organization"),
    "google deepmind": ("Google DeepMind", "Organization"),
    "anthropic": ("Anthropic", "Organization"),
    "meta ai": ("Meta AI", "Organization"),
    "hugging face": ("Hugging Face", "Organization"),
    "huggingface": ("Hugging Face", "Organization"),
    "mistral": ("Mistral AI", "Organization"),
    "cohere": ("Cohere", "Organization"),
    "nvidia": ("NVIDIA", "Organization"),

    # Models
    "gpt-4": ("GPT-4", "Model"),
    "gpt-3": ("GPT-3", "Model"),
    "gpt-3.5": ("GPT-3.5", "Model"),
    "claude": ("Claude", "Model"),
    "llama": ("LLaMA", "Model"),
    "gemini": ("Gemini", "Model"),
    "mistral 7b": ("Mistral 7B", "Model"),
    "bert": ("BERT", "Model"),
    "t5": ("T5", "Model"),

    # Conferences & Journals
    "neurips": ("NeurIPS", "Conference"),
    "icml": ("ICML", "Conference"),
    "iclr": ("ICLR", "Conference"),
    "acl": ("ACL", "Conference"),
    "emnlp": ("EMNLP", "Conference"),
    "arxiv": ("arXiv", "Journal/Preprint"),
    "jmlr": ("JMLR", "Journal/Preprint"),
    "nature": ("Nature", "Journal/Preprint"),

    # Healthcare
    "clinical trial": ("Clinical Trial", "Healthcare"),
    "ehr": ("Electronic Health Records", "Healthcare"),
    "electronic health records": ("Electronic Health Records", "Healthcare"),
    "radiology": ("Radiology", "Healthcare"),
    "genomics": ("Genomics", "Healthcare"),
    "drug discovery": ("Drug Discovery", "Healthcare"),
    "nejm": ("NEJM", "Healthcare Journal"),
    "the lancet": ("The Lancet", "Healthcare Journal"),
    "jama": ("JAMA", "Healthcare Journal"),
}


def lookup(text: str) -> list[tuple[str, str]]:
    """
    Scan text for gazetteer terms.
    Returns list of (canonical_name, category) tuples — deduplicated.
    """
    text_lower = text.lower()
    seen: set[str] = set()
    hits: list[tuple[str, str]] = []

    # Sort by length descending so longer phrases match before sub-phrases
    for term in sorted(GAZETTEER, key=len, reverse=True):
        if term in text_lower:
            canonical, category = GAZETTEER[term]
            if canonical not in seen:
                seen.add(canonical)
                hits.append((canonical, category))

    return hits
