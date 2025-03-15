import gym
import numpy

class MetaLMV3Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Sequence(gym.spaces.Discrete(16))
        self.observation_space = gym.spaces.Sequence(gym.spaces.Discrete(16))
    
    def set_task(self, task):
        self.vocabulary = task["vocabulary"]
        self.embedding = task["embedding"]
        self.hidden = task["hidden"]
        self.lm = task["lm"]
        self.action_space = gym.spaces.Sequence(gym.spaces.Discrete(self.vocabulary))
        self.observation_space = gym.spaces.Sequence(gym.spaces.Discrete(self.vocabulary))
        self.task_set = True

    def reset(self):
        if(self.task_set == False):
            raise Exception("Task not set")
        self.cached_query = self.lm.generate_query()
        return self.cached_query
    
    def step(self, action, cached=False):
        label, ppl = self.lm.label_answer(list(action))
        _, ppl_min = self.lm.generate_answer_greedy()
        _, ppl_max = self.lm.generate_answer_low()
        r = (ppl_max - ppl_min) / max(ppl - ppl_min + 0.1, 1.0e-3) - 2.0
        if(not cached):
            s = self.lm.generate_query()
        else:
            s = self.cached_query
        return tuple(s), r, False, {"label": label}
    
    def policy(self, T=1.0):
        ans, _ = self.lm.generate_answer_softmax(T=T)
        return list(ans)