from typing import List, Dict
from llama_index.core import Settings

class ProblemDecomposer:
    """Decomposes complex queries into sub-problems for better analysis."""
    
    def __init__(self):
        self.decomposition_prompt = """Analyze the following financial query and break it down into sub-problems.
        For each sub-problem, identify:
        1. The specific aspect being asked
        2. Any dependencies on other sub-problems
        3. Required information or context
        
        Query: {query}
        
        Provide the decomposition in a structured format."""
    
    def decompose_query(self, query: str) -> Dict:
        """Decompose a query into sub-problems using LLM."""
        if not Settings.llm:
            return {
                'original_query': query,
                'sub_problems': [{'query': query, 'dependencies': [], 'required_info': []}]
            }
            
        try:
            # Get LLM decomposition
            prompt = self.decomposition_prompt.format(query=query)
            response = Settings.llm.complete(prompt)
            decomposition = response.text.strip()
            print("query after decomposition", query)
            # Parse the decomposition into structured format
            sub_problems = self._parse_decomposition(decomposition)
            
            return {
                'original_query': query,
                'sub_problems': sub_problems,
                'decomposition_analysis': decomposition
            }
            
        except Exception as e:
            # Fallback to simple decomposition
            return {
                'original_query': query,
                'sub_problems': [{'query': query, 'dependencies': [], 'required_info': []}],
                'error': str(e)
            }
    
    def _parse_decomposition(self, decomposition: str) -> List[Dict]:
        """Parse the LLM decomposition into structured format."""
        # Basic parsing - can be enhanced with more sophisticated parsing
        sub_problems = []
        current_problem = None
        
        for line in decomposition.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Sub-problem'):
                if current_problem:
                    sub_problems.append(current_problem)
                current_problem = {
                    'query': line.split(':', 1)[1].strip(),
                    'dependencies': [],
                    'required_info': []
                }
            elif line.startswith('Dependencies:'):
                if current_problem:
                    current_problem['dependencies'] = [
                        dep.strip() for dep in line.split(':', 1)[1].strip().split(',')
                        if dep.strip()
                    ]
            elif line.startswith('Required Info:'):
                if current_problem:
                    current_problem['required_info'] = [
                        info.strip() for info in line.split(':', 1)[1].strip().split(',')
                        if info.strip()
                    ]
        
        if current_problem:
            sub_problems.append(current_problem)
            
        return sub_problems 