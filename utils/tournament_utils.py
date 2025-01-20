import random
from typing import Dict, List, Optional, Tuple, Any, Callable
from utils.benchmark_utils import remove_inst_tokens

class Tournament:
    """Manages solution tournaments and generates judge training examples"""
    
    def __init__(self, judge_agent, logger=None):
        """
        Initialize tournament manager
        Args:
            judge_agent: Agent that can compare solutions
            logger: Optional logger for tournament progress
        """
        self.judge_agent = judge_agent
        self.logs = []
        self.logger = logger
        
    def _log(self, message: str):
        """Log message to both internal logs and logger if available"""
        self.logs.append(message)
        if self.logger:
            self.logger.append(message)
            
    async def _run_match(
        self,
        problem: str,
        sol_a: Tuple[Any, bool, str],
        sol_b: Tuple[Any, bool, str],
        get_content: Callable = lambda x: x[0]
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Run a single tournament match between two solutions
        Args:
            problem: Problem statement
            sol_a: First solution tuple (content, is_correct, prompt)
            sol_b: Second solution tuple (content, is_correct, prompt) 
            get_content: Function to extract content from solution tuple
        Returns:
            winner: 'A' or 'B' indicating which solution won
            training_example: Optional training example if judge was wrong
        """
        try:
            # Unpack solutions
            sol_a_text = get_content(sol_a)
            sol_b_text = get_content(sol_b)
            is_correct_a = sol_a[1]
            is_correct_b = sol_b[1]
            prompt_a = sol_a[2]
            prompt_b = sol_b[2]
            
            # Get judge's decision
            judge_response = await self.judge_agent.compare_solutions(
                problem,
                sol_a_text,
                sol_b_text
            )
            
            # Parse response
            response = judge_response.strip().upper()
            if response and response[0] in ['A', 'B']:
                winner = response[0]
            else:
                winner = random.choice(['A', 'B'])
                self._log(f"Invalid judge response, randomly chose {winner}")
            
            # Generate training example if judge was wrong
            training_example = None
            if is_correct_a != is_correct_b:  # One correct, one incorrect
                judge_chose_correct = (winner == 'A' and is_correct_a) or (winner == 'B' and is_correct_b)
                if not judge_chose_correct:
                    # Split solutions into steps
                    correct_sol = sol_a_text if is_correct_a else sol_b_text
                    wrong_sol = sol_b_text if is_correct_a else sol_a_text
                    
                    # Split solutions for judge
                    correct_steps = split_into_steps(remove_inst_tokens(correct_sol))
                    wrong_steps = split_into_steps(wrong_sol)
                    
                    # Remove last step from both solutions
                    truncated_correct = "\n\n".join(correct_steps[:-1]) if len(correct_steps) > 1 else correct_steps[0]
                    truncated_wrong = "\n\n".join(wrong_steps[:-1]) if len(wrong_steps) > 1 else wrong_steps[0]
                    
                    winner_prompt = prompt_a if winner == 'A' else prompt_b
                    training_example = {
                        'alignment': 'judge',
                        'type': 'solution',
                        'problem': problem,
                        'prompt': {'content': winner_prompt, 'role': 'user'},
                        'chosen': {'content': truncated_correct, 'role': 'assistant'},
                        'rejected': {'content': truncated_wrong, 'role': 'assistant'},
                        'score_chosen': 1.0,
                        'score_rejected': 0.0
                    }
                    
            return winner, training_example
            
        except Exception as e:
            self._log(f"Error in tournament match: {str(e)}")
            return None, None
            
    async def run_tournament(
        self,
        solutions: List[Tuple[Any, bool, str]],
        problem: str,
        get_content: Callable = lambda x: x[0]
    ) -> Tuple[List[Tuple[Any, bool, str]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run tournament between solutions to rank them and generate training examples
        Args:
            solutions: List of solution tuples (content, is_correct, prompt)
            problem: Problem statement
            get_content: Function to extract content from solution tuple
        Returns:
            sorted_solutions: Solutions sorted by tournament performance
            tournament_results: List of training examples from tournament
            stats: Tournament statistics
        """
        if len(solutions) < 2:
            return solutions, [], {}
            
        wins = {i: 0 for i in range(len(solutions))}
        judge_correct = 0
        judge_total = 0
        tournament_results = []
        
        # Run round-robin tournament
        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                winner, training_example = await self._run_match(
                    problem, 
                    solutions[i], 
                    solutions[j],
                    get_content
                )
                
                if winner:
                    winner_idx = i if winner == 'A' else j
                    wins[winner_idx] += 1
                    
                    # Track judge accuracy
                    if solutions[i][1] != solutions[j][1]:  # Different correctness
                        judge_total += 1
                        if (winner == 'A' and solutions[i][1]) or (winner == 'B' and solutions[j][1]):
                            judge_correct += 1
                            
                if training_example:
                    tournament_results.append(training_example)
                    
        # Sort solutions by wins
        sorted_indices = sorted(wins.keys(), key=lambda x: wins[x], reverse=True)
        sorted_solutions = [solutions[i] for i in sorted_indices]
        
        # Calculate stats
        stats = {
            'judge_accuracy': judge_correct/judge_total if judge_total > 0 else 0,
            'judge_decisions': judge_total,
            'solution_ranking': [1 if sol[1] else 0 for sol in sorted_solutions]
        }
        
        # Log results
        self._log("\n=== Tournament Results ===")
        if judge_total > 0:
            self._log(f"Judge accuracy: {judge_correct}/{judge_total} ({stats['judge_accuracy']*100:.1f}% correct)")
        self._log(f"Solution ranking (1=correct, 0=incorrect): {stats['solution_ranking']}")
        
        return sorted_solutions, tournament_results, stats
