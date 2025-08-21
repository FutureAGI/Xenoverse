if __name__ == "__main__":
    import numpy as np
    import pickle 
    import time
    from xenoverse.anyhvac.anyhvac_env_vis import HVACEnvVisible, HVACEnv
    from xenoverse.anyhvac.anyhvac_sampler import HVACTaskSampler
    from xenoverse.anyhvac.anyhvac_solver import HVACSolverGTPID

    pid_type = "HVACSolverGTPID" #"temperarure" , "HVACSolverGTPID"
    env = HVACEnvVisible(verbose=True)
    print("Sampling hvac tasks...")

    TASK_CONFIG_PATH = "./gen_env/hvac_task.pkl"
    

    try:
        with open(TASK_CONFIG_PATH, "rb") as f:
            task = pickle.load(f)
        print(f"Loaded existing task config from {TASK_CONFIG_PATH}")
    
    except FileNotFoundError:
        print("Sampling new HVAC tasks...")
        task = HVACTaskSampler(control_type='Temperature')
        with open(TASK_CONFIG_PATH, "wb") as f:
            pickle.dump(task, f)
        print(f"... Saved new task config to {TASK_CONFIG_PATH}")

    print("... Finished Sampling")
    env.set_task(task)
    terminated, truncated = False,False
    obs, info = env.reset()
    agent = HVACSolverGTPID(env)
    while (not terminated) and (not truncated):

        if pid_type == "temperarure":
            action = env._pid_action()
        elif pid_type == "HVACSolverGTPID":
            action = agent.policy(obs)
            if action.shape != env.action_space.shape:
                print(f"Warning: Action shape from agent ({action.shape}) does not match env action space shape ({env.action_space.shape}).")
               
                if action.ndim == 2 and action.shape[0] == 1 and action.shape[1] == env.action_space.shape[0]:
                    action = action.squeeze(0)
                elif action.size != env.action_space.shape[0] * env.action_space.shape[1] if len(env.action_space.shape) > 1 else env.action_space.shape[0] :
                    print(f"Action size mismatch: {action.size} vs {env.action_space.shape}")
        obs, reward, terminated, truncated, info = env.step(action)
        cool_power = round(np.mean(info.get("cool_power", 0)),4)
        heat_power = round(np.mean(info.get("heat_power", 0)),4)
        info_total = f"energy_cost: {round(info.get('energy_cost', 0),4)}, target_cost: {round(info.get('target_cost', 0),4)}, switch_cost: {round(info.get('switch_cost', 0),4)},cool_power: {cool_power}, heat_power: {heat_power}"
