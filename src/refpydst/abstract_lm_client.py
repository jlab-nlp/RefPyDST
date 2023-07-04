import abc
from typing import List, Dict, Callable


class AbstractLMClient(metaclass=abc.ABCMeta):
    """
    Any 'client' implementing these methods should be able to support a RefPyDST experiment as the generative LM
    """

    @abc.abstractmethod
    def __init__(self, stop_sequences: List[str] = None, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def greedy_lm_completion(self, prompt_text: str) -> Dict[str, float]:
        """
        Given a prompt, generate a completion using the given engine and other completion parameters.

        :param prompt_text: prefix text for OpenAI Completion API
        :return: the single most likely completion for the prompt (greedily sampled), not including the prompt tokens.
        """
        pass

    @abc.abstractmethod
    def top_p_lm_completion(self, prompt_text: str, top_p: float = 0.9, n: int = 5, best_of: int = 10,
                            max_tokens: int = 120, **kwargs) -> Dict[str, float]:
        """
        Given a prompt, generate a completion using the given engine and other completion parameters.

        :param prompt_text: prefix text for OpenAI Completion API
        :return: the single most likely completion for the prompt (greedily sampled), not including the prompt tokens.
        """
        pass

    @abc.abstractmethod
    def get_completion_log_probabilities(self, prompt_text: str, completion: str,
                                         token_log_probs_telemetry_hook: Callable[[List[float]], None] = None) -> List[
        float]:
        pass
