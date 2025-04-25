import gymnasium as gym
import numpy
import xenoverse.anymdp
from xenoverse.anymdp import  AnyMDPSolverOpt, AnyMDPTaskSampler
from xenoverse.anymdp import anymdp_task_visualizer
from xenoverse.anymdp.solver import update_value_matrix


def run_rnd_opt(task, max_steps=10000, prt_freq=1000):
    # Test Random Policy
    env = gym.make("anymdp-v0")
    env.set_task(task)
    state, info = env.reset()

    acc_reward_rnd = 0
    epoch_reward_rnd = 0
    epoch_step_rnd = 0

    steps = 0
    while steps < max_steps:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        acc_reward_rnd += reward
        epoch_reward_rnd += reward
        steps += 1
        epoch_step_rnd += 1

        if(steps % prt_freq == 0 and steps > 0):
            print("Step:{}\tEpoch Reward: {}".format(steps, epoch_reward_rnd))
            epoch_reward_rnd = 0
        if(terminated or truncated):
            state, info = env.reset()
            epoch_step_rnd = 0
    print("Random Policy Summary: {}, Average Length: {}".format(acc_reward_rnd, epoch_step_rnd))

    # Test AnyMDPSolverOpt
    solver = AnyMDPSolverOpt(env)
    state, info = env.reset()

    acc_reward_opt = 0
    epoch_reward_opt = 0
    epoch_step_opt = 0
    epoch_steps_opt = []
    epoch_trajectory = [env.inner_state]

    steps = 0
    while steps < max_steps:
        action = solver.policy(state)
        state, reward, terminated, truncated, info = env.step(action)
        epoch_trajectory.append((env.inner_state, int(action)))
        acc_reward_opt += reward
        epoch_reward_opt += reward
        steps += 1
        epoch_step_opt += 1
        if(steps % prt_freq == 0 and steps > 0):
            print("Step:{}\tEpoch Reward: {}".format(steps, epoch_reward_opt))
            epoch_reward_opt = 0
        if(terminated or truncated):
            state, info = env.reset()
            epoch_trajectory.append(env.inner_state)
            epoch_steps_opt.append(epoch_step_opt)
            epoch_step_opt = 0
    print("Optimal Solver Summary:  {}, Averge Length: {}, Epoch Trajectory: {}, Stop States: {}".format(acc_reward_opt, numpy.mean(epoch_steps_opt), epoch_trajectory[-50:], env.s_e))

    if(acc_reward_opt < acc_reward_rnd + 0.10):
        print('!'* 50, 'solution is abnormal', '!'* 50)
        return False
    else:
        return True

if __name__=="__main__":
    ns = 8
    na = 5
    while True:
        task = AnyMDPTaskSampler(state_space=ns, 
                             action_space=na,
                             min_state_space=None,
                             verbose=True)
        if(not run_rnd_opt(task)):
            break
    env = gym.make("anymdp-v0")
    env.set_task(task)
    solver = AnyMDPSolverOpt(env)
    print("value matrix:", solver.value_matrix)
    print("position reward:", numpy.mean(env.reward, axis=(0, 1)))
    print("optimal solution:", numpy.argmax(solver.value_matrix, axis=1))
    vm_rnd = numpy.zeros((ns, na))
    vm_rnd = update_value_matrix(env.transition, 
                        env.reward, 
                        0.99, 
                        vm_rnd, is_greedy=False)
    print("value matrix random:", vm_rnd)
    print("random solution:", numpy.argmax(vm_rnd, axis=1))

    anymdp_task_visualizer(task)