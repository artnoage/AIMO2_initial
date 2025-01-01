import asyncio
from typing import Dict, List, Optional, Tuple, Any
from bench_utils.benchmark_utils import (
    validate_analysis, 
    validate_step,
    extract_answer_from_solution,
    NumericVerifier
)
from bench_utils.agents import (
    AnalysisAgent,
    NextStepAgent,
    CompletionAgent
)

class ListGenerator:
    """Generates lists of solution components with best/worst variants"""
    
    def __init__(self, solver, best_of: int, completions: int):
        self.solver = solver
        self.best_of = best_of
        self.completions = completions
        self.analysis_agent = AnalysisAgent(solver)
        self.step_agent = NextStepAgent(solver)
        self.completion_agent = CompletionAgent(solver)
        self.verifier = NumericVerifier()

    async def _score_with_completions(
        self,
        problem: str,
        current_solution: str,
        correct_answer: str
    ) -> float:
        """Score a partial solution by attempting completions"""
        successful = 0
        
        for _ in range(self.completions):
            try:
                complete_solution = current_solution + await self.completion_agent.generate(
                    problem,
                    current_solution
                )
                score, total_steps, _ = await self.verifier.verify(
                    complete_solution,
                    correct_answer,
                    problem
                )
                if score == total_steps:
                    successful += 1
            except Exception:
                continue
                
        return successful / self.completions

    async def generate(
        self,
        problem: str,
        correct_answer: str
    ) -> List[Dict[str, Any]]:
        """
        Generate solution components with best/worst variants.
        Returns list of dicts with prompt/chosen/rejected/scores.
        """
        results = []
        current_solution = ""
        
        # Generate and score analyses
        analyses = []
        analysis_prompt = None
        
        for _ in range(self.best_of):
            try:
                if analysis_prompt is None:
                    prompt, analysis = await self.analysis_agent.generate(
                        problem,
                        return_prompt=True
                    )
                    analysis_prompt = prompt
                else:
                    analysis = await self.analysis_agent.generate(problem)
                
                is_valid, _ = validate_analysis(analysis)
                if is_valid:
                    score = await self._score_with_completions(
                        problem,
                        analysis,
                        correct_answer
                    )
                    analyses.append((analysis, score))
            except Exception:
                continue
                
        if len(analyses) < 2:
            return []
            
        # Sort and get best/worst analysis
        analyses.sort(key=lambda x: x[1])
        results.append({
            'prompt': {'content': analysis_prompt, 'role': 'user'},
            'chosen': {'content': analyses[-1][0], 'role': 'assistant'},
            'rejected': {'content': analyses[0][0], 'role': 'assistant'},
            'score_chosen': analyses[-1][1],
            'score_rejected': analyses[0][1]
        })
        
        # Use best analysis as starting point
        current_solution = analyses[-1][0]
        step_num = 1
        
        while True:
            steps = []
            step_prompt = None
            
            # Generate and score steps
            for _ in range(self.best_of):
                try:
                    if step_prompt is None:
                        prompt, step = await self.step_agent.generate(
                            problem,
                            current_solution,
                            return_prompt=True
                        )
                        step_prompt = prompt
                    else:
                        step = await self.step_agent.generate(
                            problem,
                            current_solution
                        )
                    
                    test_solution = current_solution + step
                    
                    # Check if step contains answer
                    answer = extract_answer_from_solution(test_solution)
                    if answer is not None:
                        score, _, _ = await self.verifier.verify(
                            test_solution,
                            correct_answer,
                            problem
                        )
                        if score:  # Valid answer found
                            steps.append((step, 1.0))
                            break
                            
                    # Score step if no answer yet
                    if validate_step(step):
                        score = await self._score_with_completions(
                            problem,
                            test_solution,
                            correct_answer
                        )
                        steps.append((step, score))
                except Exception:
                    continue
                    
            if not steps:  # No valid steps generated
                break
                
            # Sort and get best/worst step
            steps.sort(key=lambda x: x[1])
            results.append({
                'prompt': {'content': step_prompt, 'role': 'user'},
                'chosen': {'content': steps[-1][0], 'role': 'assistant'},
                'rejected': {'content': steps[0][0], 'role': 'assistant'},
                'score_chosen': steps[-1][1],
                'score_rejected': steps[0][1]
            })
            
            # Use best step and continue
            current_solution += steps[-1][0]
            
            # Check if we found a valid answer
            if extract_answer_from_solution(current_solution) is not None:
                break
                
            step_num += 1
            
        return results
