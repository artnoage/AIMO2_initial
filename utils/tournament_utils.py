import random
from typing import Dict, List, Optional, Tuple, Any, Callable
from utils.benchmark_utils import remove_inst_tokens, split_into_steps

class Tournament:
    """Manages solution tournaments and generates judge training examples"""
    
    JUDGE_PROMPT_TEMPLATE = (
        "You are a mathematics judge. You will be presented with a problem and two proposed partial or full solutions: "
        "Solution A and Solution B. Your task is to thoroughly evaluate both solutions and determine which one "
        "demonstrates stronger reasoning and is more likely to be correct.\n\n"
        "Problem:\n{problem}\n\n"
        "Solution A:\n{solution_a}\n\n"
        "Solution B:\n{solution_b}\n\n"
        "Which solution is better, A or B?"
    )
    
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
        correct_answer: str,
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

            
            # Get judge's prompt and decision
            judge_prompt, judge_response = await self.judge_agent.compare_solutions(
                problem,
                sol_a_text,
                sol_b_text,
                return_prompt=True
            )
            
            # Parse response
            response = judge_response.strip().upper()
            if response and response[0] in ['A', 'B']:
                winner = response[0]
            else:
                winner = random.choice(['A', 'B'])
                self._log(f"Invalid judge response, randomly chose {winner}")
            
            self._log(f"\nJudge Decision:")
            self._log(f"Solution A correct: {is_correct_a}")
            self._log(f"Solution B correct: {is_correct_b}")
            self._log(f"Judge chose: {winner}")
            
            # Generate training example if judge was wrong
            training_example = None
            if is_correct_a != is_correct_b:  # One correct, one incorrect
                judge_chose_correct = (winner == 'A' and is_correct_a) or (winner == 'B' and is_correct_b)
                self._log(f"Judge chose {'correctly' if judge_chose_correct else 'incorrectly'}")
                self._log(f"Winner {winner} was {'correct' if ((winner == 'A' and is_correct_a) or (winner == 'B' and is_correct_b)) else 'incorrect'}")
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
                    
                    
                    # Create judge prompt using template with truncated solutions
                    judge_prompt = Tournament.JUDGE_PROMPT_TEMPLATE.format(
                        problem=problem,
                        solution_a=truncated_correct if is_correct_a else truncated_wrong,
                        solution_b=truncated_wrong if is_correct_a else truncated_correct
                    )
                    
                    # Create judge training example based on which solution was correct
                    training_example = {
                        'data_type': 'training',
                        'alignment': 'judge',
                        'type': 'full_solution',
                        'problem': problem,
                        'correct_answer': correct_answer,
                        'prompt': {'content': judge_prompt, 'role': 'user'}, 
                        'chosen': {'content': 'A' if is_correct_a else 'B', 'role': 'assistant'},
                        'rejected': {'content': 'B' if is_correct_a else 'A', 'role': 'assistant'},
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
        correct_answer: str,
        get_content: Callable = lambda x: x[0]
    ) -> Tuple[List[Tuple[Any, bool, str]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run single-elimination bracket tournament between solutions
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
            
        # Initialize tracking variables
        judge_correct = 0
        judge_total = 0
        tournament_results = []
        eliminated = set()
        
        # Create initial random seeding
        current_bracket = list(enumerate(solutions))
        random.shuffle(current_bracket)
        
        # Track solution performance
        solution_wins = {i: 0 for i in range(len(solutions))}
        
        # Run tournament rounds until winner is determined
        round_num = 1
        while len(current_bracket) > 1:
            self._log(f"\n=== Round {round_num} ===")
            next_bracket = []
            
            # Pair up solutions and run matches
            for i in range(0, len(current_bracket), 2):
                if i + 1 >= len(current_bracket):
                    # Bye round - solution automatically advances
                    next_bracket.append(current_bracket[i])
                    continue
                    
                idx_a, sol_a = current_bracket[i]
                idx_b, sol_b = current_bracket[i + 1]
                
                winner, training_example = await self._run_match(
                    problem,
                    correct_answer,
                    sol_a,
                    sol_b,
                    get_content
                )
                
                if winner:
                    winner_idx, winner_sol = (idx_a, sol_a) if winner == 'A' else (idx_b, sol_b)
                    solution_wins[winner_idx] += 1
                    next_bracket.append((winner_idx, winner_sol))
                    eliminated.add(idx_b if winner == 'A' else idx_a)
                    
                    # Track judge accuracy
                    if sol_a[1] != sol_b[1]:  # Different correctness
                        judge_total += 1
                        if (winner == 'A' and sol_a[1]) or (winner == 'B' and sol_b[1]):
                            judge_correct += 1
                            
                if training_example:
                    tournament_results.append(training_example)
            
            current_bracket = next_bracket
            round_num += 1
        
        # Sort solutions by tournament performance and elimination order
        sorted_indices = sorted(
            range(len(solutions)),
            key=lambda x: (solution_wins[x], -1 if x not in eliminated else list(eliminated).index(x)),
            reverse=True
        )
        sorted_solutions = [solutions[i] for i in sorted_indices]
        
        # Calculate stats
        stats = {
            'judge_accuracy': judge_correct/judge_total if judge_total > 0 else 0,
            'judge_decisions': judge_total,
            'solution_ranking': [1 if sol[1] else 0 for sol in sorted_solutions]
        }
        
        # Log results
        self._log("\n=== Tournament Complete ===")
        if judge_total > 0:
            self._log(f"Judge decisions made: {judge_total}")
            self._log(f"Judge correct decisions: {judge_correct}/{judge_total} ({(judge_correct/judge_total)*100:.1f}%)")
            
        self._log("\nRanked Solutions:")
        for i, (solution, is_correct, _) in enumerate(sorted_solutions, 1):
            status = "✓ Correct" if is_correct else "✗ Incorrect"
            self._log(f"{i}. {status}")
        
        return sorted_solutions, tournament_results, stats
