import numpy
from xenoverse.anymdp import  AnyMDPSolverOpt, AnyMDPSolverMBRL, AnyMDPSolverQ

class RandomAgent:
    def __init__(self, env):
        self.na = env.na
        self.ns = env.ns
        self.action_space = env.action_space

    def policy(self, state, is_test=False):
        return self.action_space.sample()
    
    def learner(self, s, a, ns, r, terminated, truncated):
        pass

def train(env, max_epochs=1000, 
          gamma=0.99, 
          solver_type='q', 
          lr=0.20, 
          c=0.05,
          test_interval=100,
          test_epochs=3,
          is_verbose=True): 
        # Test AnyMDPSolverQ
        if(solver_type.lower()=='q'):
            solver = AnyMDPSolverQ(env, gamma=gamma, alpha=lr)
        elif(solver_type.lower()=='mbrl'):
            solver = AnyMDPSolverMBRL(env, gamma=gamma, c=c)
        elif(solver_type.lower()=='opt'):
            solver = AnyMDPSolverOpt(env, gamma=gamma)
        elif(solver_type.lower()=='random'):
            solver = RandomAgent(env)
        else:
            raise ValueError('Invalid Solver Type')

        epoch_rewards = []
        epoch_steps = []
        epoch_test_rewards = []

        epochs = 0

        def epoch_run(is_test=False, epochs=1):
            rewards = []
            steps = []
            for epoch in range(epochs):
                state, info = env.reset()
                terminated, truncated = False, False
                epoch_reward = 0
                epoch_step = 0
                while not terminated and not truncated:
                    action = solver.policy(state, is_test=is_test)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    solver.learner(state, action, next_state, reward, terminated, truncated)
                    epoch_reward += reward
                    epoch_step += 1
                    state = next_state
                rewards.append(epoch_reward)
                steps.append(epoch_step)

            return numpy.mean(rewards), numpy.mean(steps)

        while epochs < max_epochs:
            train_rewards, train_steps = epoch_run(epochs=test_interval)
            test_rewards, test_steps = epoch_run(is_test=True, epochs=test_epochs)
            epoch_rewards.append(train_rewards)
            epoch_steps.append(train_steps)
            epoch_test_rewards.append(test_rewards)
            epochs += test_interval
            if(is_verbose):
                print("[{}]-Run\tEpoch:{}\tMean Train Epoch Reward: {:.2f}\tMean Test Epoch Reward: {:.2f}\tMean Steps In Epoch: {:.2f}\t".format(solver_type, epochs, epoch_rewards[-1], epoch_test_rewards[-1], epoch_steps[-1]))
        print("Solver Summary: {:.3f}".format(numpy.mean(test_rewards)))

        return epoch_rewards, epoch_test_rewards, epoch_steps