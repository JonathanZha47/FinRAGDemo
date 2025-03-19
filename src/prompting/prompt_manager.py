from typing import Dict, List, Optional
from src.config.config_loader import ConfigLoader

class PromptManager:
    """Manages different prompt engineering techniques and verification."""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config = config_loader
        
        # Few-shot examples for financial advice
        self.few_shot_examples = [
            {
                "question": "What's the difference between a Roth IRA and Traditional IRA?",
                "answer": "A Traditional IRA offers tax-deductible contributions but taxable withdrawals in retirement, while a Roth IRA has after-tax contributions but tax-free qualified withdrawals. Traditional IRAs require minimum distributions (RMDs) starting at age 72, while Roth IRAs don't have RMDs."
            },
            {
                "question": "How should I diversify my investment portfolio?",
                "answer": "A well-diversified portfolio typically includes: 1) Different asset classes (stocks, bonds, cash), 2) Various sectors (technology, healthcare, etc.), 3) Geographic diversity (domestic and international), 4) Different market capitalizations (large-cap, mid-cap, small-cap). The specific allocation depends on your risk tolerance and time horizon."
            }
        ]
        
        # Persona definitions
        self.personas = {
            "financial_advisor": "You are a licensed financial advisor with 15 years of experience, specializing in personal finance and retirement planning. You always consider the client's best interests and adhere to fiduciary responsibilities.",
            "risk_analyst": "You are a risk management specialist with expertise in analyzing market risks and compliance requirements. You focus on identifying potential risks and ensuring regulatory compliance.",
            "investment_strategist": "You are an investment strategist with deep knowledge of market dynamics and portfolio management. You specialize in creating balanced, long-term investment strategies."
        }
        
    def create_few_shot_prompt(self, query: str) -> str:
        """Create a prompt using few-shot learning approach."""
        prompt = "Here are some examples of financial advice questions and answers:\n\n"
        
        for example in self.few_shot_examples:
            prompt += f"Question: {example['question']}\n"
            prompt += f"Answer: {example['answer']}\n\n"
            
        prompt += f"Now, please answer this question in a similar style:\n"
        prompt += f"Question: {query}\n"
        prompt += "Answer:"
        
        return prompt
        
    def create_context_prompt(self, query: str, context: Optional[List[str]] = None) -> str:
        """Create a prompt using context-based approach."""
        prompt = "Using the following context, provide a detailed and accurate answer:\n\n"
        
        if context:
            prompt += "Context:\n"
            for ctx in context:
                prompt += f"- {ctx}\n"
            
        prompt += f"\nQuestion: {query}\n"
        prompt += "Provide a comprehensive answer based on the given context. If any information is not supported by the context, explicitly state so."
        
        return prompt
        
    def create_cot_prompt(self, query: str) -> str:
        """Create a prompt using chain-of-thought reasoning."""
        prompt = "Let's approach this financial question step by step:\n\n"
        prompt += f"Question: {query}\n\n"
        prompt += "1. First, let's identify the key financial concepts involved.\n"
        prompt += "2. Then, let's analyze how these concepts relate to the question.\n"
        prompt += "3. Next, let's consider any relevant regulations or guidelines.\n"
        prompt += "4. Finally, let's formulate a comprehensive answer.\n\n"
        prompt += "Please follow this reasoning process and provide a detailed explanation:"
        
        return prompt
        
    def create_persona_prompt(self, query: str, persona: str) -> str:
        """Create a prompt using persona-based approach."""
        if persona not in self.personas:
            persona = "financial_advisor"  # default persona
            
        prompt = f"{self.personas[persona]}\n\n"
        prompt += "Given your expertise and role, please provide advice on the following question:\n"
        prompt += f"{query}\n\n"
        prompt += "Ensure your response reflects your professional perspective and expertise."
        
        return prompt
        
    def verify_citations(self, response: str, context: List[str]) -> Dict:
        """Verify if the response content is supported by the provided context."""
        citations = {
            "verified": [],
            "unverified": [],
            "verification_score": 0.0
        }
        
        # Split response into statements
        statements = response.split(". ")
        
        for statement in statements:
            statement = statement.strip()
            if not statement:
                continue
                
            # Check if statement is supported by context
            found = False
            for ctx in context:
                if any(sent.lower() in ctx.lower() for sent in [statement]):
                    citations["verified"].append({
                        "statement": statement,
                        "source": ctx[:100] + "..."  # First 100 chars as reference
                    })
                    found = True
                    break
                    
            if not found:
                citations["unverified"].append(statement)
                
        # Calculate verification score
        total_statements = len(statements)
        verified_statements = len(citations["verified"])
        citations["verification_score"] = verified_statements / total_statements if total_statements > 0 else 0.0
        
        return citations
        
    def detect_hallucination(self, response: str, context: List[str], citations: Dict) -> Dict:
        """Detect potential hallucinations in the response."""
        hallucination_check = {
            "potential_hallucinations": [],
            "confidence_score": 0.0,
            "warning_flags": []
        }
        
        # Check for statements without citations
        hallucination_check["potential_hallucinations"] = citations["unverified"]
        
        # Calculate confidence score based on citation verification
        hallucination_check["confidence_score"] = citations["verification_score"]
        
        # Add warning flags for specific cases
        if citations["verification_score"] < 0.7:
            hallucination_check["warning_flags"].append(
                "Low citation verification score - significant content may be unsupported"
            )
            
        if len(citations["unverified"]) > len(citations["verified"]):
            hallucination_check["warning_flags"].append(
                "More unverified statements than verified ones"
            )
            
        return hallucination_check
        
    def apply_guardrails(self, response: str, sec_guidelines: List[str]) -> Dict:
        """Check if response complies with SEC guidelines."""
        compliance_check = {
            "compliant": True,
            "violations": [],
            "warnings": []
        }
        
        # Check against each SEC guideline
        for guideline in sec_guidelines:
            # Add your specific SEC guideline checks here
            # This is a placeholder for the actual implementation
            pass
            
        return compliance_check

    def get_response(self, query: str, context: Optional[List[str]] = None, prompt_type: str = "context", persona: Optional[str] = None, provider_name: str = "openai") -> str:
        """Generate a response using the specified prompt engineering technique."""
        # Create the appropriate prompt based on the prompt type
        if prompt_type == "few-shot":
            prompt = self.create_few_shot_prompt(query)
        elif prompt_type == "context":
            prompt = self.create_context_prompt(query, context)
        elif prompt_type == "chain-of-thought":
            prompt = self.create_cot_prompt(query)
        elif prompt_type == "persona":
            if not persona:
                persona = "financial_advisor"
            prompt = self.create_persona_prompt(query, persona)
        else:
            # Default to context-based prompt
            prompt = self.create_context_prompt(query, context)

        try:
            # Get the LLM provider using the same approach as in utils.py
            from llm_providers import get_llm_provider
            
            llm = get_llm_provider(provider_name)
            if llm is None:
                raise ValueError(f"Failed to initialize LLM provider: {provider_name}")
            
            # Add system prompt for better context
            system_prompt = (
                "You are an AI financial advisor. Provide clear, factual financial advice based "
                "on the given context. Always be transparent about limitations and risks. "
                "If you're unsure about something, say so explicitly."
            )
            
            # Generate response using the provider
            response = llm.generate_response(
                prompt,
                system_prompt=system_prompt,
                temperature=0.1  # Using a lower temperature for more focused responses
            )
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}" 