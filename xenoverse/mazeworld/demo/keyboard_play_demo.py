import gymnasium as gym
import sys
import argparse
import xenoverse.mazeworld
from xenoverse.mazeworld import MazeTaskSampler

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Playing the maze world demo with your keyboard')
    parser.add_argument('--max_steps', type=int, default=1000000)
    parser.add_argument('--visibility_3D', type=float, default=12, help="3D vision range, Only valid in 3D mode")
    parser.add_argument('--save_replay', type=str, default=None, help="Save the replay trajectory in file")

    args = parser.parse_args()

    # Create the environment
    maze_env = gym.make("mazeworld-v2", 
                        enable_render=True, 
                        max_steps=args.max_steps, 
                        visibility_3D=args.visibility_3D, 
                        command_in_observation=False,
                        render_scale=320)

    # Sample and set the task
    task = MazeTaskSampler()
    maze_env.set_task(task)

    maze_env.reset()
    terminated, truncated = False, False
    sum_reward = 0

    while (not terminated) and (not truncated):
        maze_env.render()
        state, reward, terminated, truncated, _ = maze_env.step(None)
        sum_reward += reward
        print("Instant r = %.2f, Accumulate r = %.2f" % (reward, sum_reward))
        if(maze_env.key_done):
            break
    print("Episode is over! You got %.2f score."%sum_reward)

    if(args.save_replay is not None):
        maze_env.save_trajectory(args.save_replay)
