import gymnasium as gym
import sys
import argparse
import time
import xenoverse.mazeworld
from xenoverse.mazeworld import MazeTaskSampler
from xenoverse.mazeworld.agents import SmartSLAMAgent, OracleAgent

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Playing the maze world demo with your keyboard')
    parser.add_argument('--max_steps', type=int, default=1000000)
    parser.add_argument('--visibility_3D', type=float, default=12, help="3D vision range, Only valid in 3D mode")
    parser.add_argument('--save_replay', type=str, default=None, help="Save the replay trajectory in file")
    parser.add_argument('--memory_keep_ratio', type=float, default=1.0, 
                        help="Keep ratio of memory when the agent switch from short to long term memory. 1.0 means perfect memory, 0.0 means no memory")
    parser.add_argument('--oracle', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=False)

    args = parser.parse_args()

    # create the environment
    maze_env = gym.make("mazeworld-v2", enable_render=False, max_steps=args.max_steps, visibility_3D=args.visibility_3D, 
                        command_in_observation=True)

    # sample the task
    task = MazeTaskSampler(verbose=True)
    maze_env.set_task(task)

    # create an smart SLAM agent
    if(args.oracle):
        agent = OracleAgent(maze_env=maze_env, render=True)
    else:
        agent = SmartSLAMAgent(maze_env=maze_env, memory_keep_ratio=args.memory_keep_ratio, render=True)

    observation, _ = maze_env.reset()
    terminated, truncated = False, False
    sum_reward = 0
    reward = 0

    while (not terminated) and (not truncated):
        action = agent.step(observation, reward)
        observation, reward, terminated, truncated, _ = maze_env.step(action)
        sum_reward += reward
        if(args.verbose):
            print("Instant r = %.2f, Accumulate r = %.2f" % (reward, sum_reward))
        if(maze_env.key_done):
            break
    print("Episode is over! You got %.2f score."%sum_reward)

    if(args.save_replay is not None):
        maze_env.save_trajectory(args.save_replay)