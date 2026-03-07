# src/agents/llm_answerer.py
"""LLM-based answer synthesis using OpenAI."""

import os
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class LLMAnswerer:
    """Generate concise answers from retrieved context using OpenAI."""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"✓ OpenAI client initialized ({model})")
            except Exception as e:
                logger.warning(f"OpenAI init failed: {e}")
        else:
            logger.warning("OPENAI_API_KEY not set - LLM answers disabled")
    
    def generate_answer(self, query: str, contexts: List[str], entity: Optional[str] = None) -> Dict:
        """Generate a concise answer from retrieved contexts."""
        if not self.client or not contexts:
            return {"answer": None, "confidence": 0.0, "error": "LLM unavailable or no context"}
        
        try:
            context_text = "\n\n".join([f"[{i+1}] {c[:500]}..." for i, c in enumerate(contexts[:5])])
            entity_instruction = f"Focus on extracting the {entity} figure if present. " if entity else ""
            
            prompt = f"""You are a financial document analyst. Answer using ONLY the provided context. Be concise.

Question: {query}
{entity_instruction}
Context excerpts:
{context_text}

Instructions:
1. If context contains specific numerical answer, state it clearly (e.g., "ETB 14.2 billion").
2. If multiple values exist, list the most relevant.
3. If context doesn't contain the answer, say so.
4. Keep answer under 3 sentences.

Answer:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150
            )
            
            answer = response.choices[0].message.content.strip()
            confidence = 0.9 if any(c.isdigit() for c in answer) else 0.7
            
            return {"answer": answer, "confidence": confidence, "model_used": self.model}
            
        except Exception as e:
            logger.error(f"LLM answer generation failed: {e}")
            return {"answer": None, "confidence": 0.0, "error": str(e)}