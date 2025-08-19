import os
import numpy as np
import pickle 
import time
from typing import List, Tuple
from pathlib import Path
from xenoverse.anyhvacv2.anyhvac_env_vis import HVACEnvVisible, HVACEnv
from xenoverse.anyhvacv2.anyhvac_sampler import HVACTaskSampler
from xenoverse.anyhvacv2.anyhvac_solver import HVACSolverGTPID


def create_and_validate_hvac_env(
    pid_type: str = "temperarure",
    task_config_path: str = "./gen_env/hvac_task_{}_time.pkl",  # 注意这里改为包含{}的格式字符串
    max_steps: int = 1000,
    num_valid_envs: int = 5,
    verbose: bool = True
) -> Tuple[List[HVACEnvVisible], List[str]]:
    """
    自动创建并验证 HVAC 环境，保留 N 个合适的环境及其配置文件路径
    
    参数:
        pid_type: PID 控制器类型 ("temperarure" 或 "HVACSolverGTPID")
        task_config_path: 任务配置保存路径的格式字符串，包含{}用于插入序号
        max_steps: 环境需要运行的最大步数
        num_valid_envs: 需要保留的有效环境数量
        verbose: 是否打印详细信息
        
    返回:
        Tuple[List[HVACEnvVisible], List[str]]: (有效环境列表, 对应的配置文件路径列表)
    """
    # 确保输出目录存在
    Path(task_config_path).parent.mkdir(parents=True, exist_ok=True)
    
    valid_envs = []
    valid_config_paths = []
    attempts = 0
    max_attempts = 100  # 防止无限循环
    
    while len(valid_envs) < num_valid_envs and attempts < max_attempts:
        attempts += 1
        current_config_path = task_config_path.format(attempts)
        
        if verbose:
            print(f"\n尝试 #{attempts} - 创建环境... (配置将保存到: {current_config_path})")
        
        try:
            # 采样新任务配置
            task = HVACTaskSampler(control_type='Temperature')
            
            # 保存任务配置
            with open(current_config_path, "wb") as f:
                pickle.dump(task, f)
            
            # 创建环境
            env = HVACEnvVisible(verbose=verbose)
            env.set_task(task)
            
            # 初始化环境
            terminated, truncated = False, False
            obs, info = env.reset()
            time.sleep(1)  # 短暂暂停
            
            # 创建控制器
            agent = HVACSolverGTPID(env) if pid_type == "HVACSolverGTPID" else None
            
            # 运行环境
            step_count = 0
            while (not terminated) and (not truncated) and (step_count < max_steps):
                if pid_type == "temperarure":
                    action = env._pid_action()
                elif pid_type == "HVACSolverGTPID":
                    action = agent.policy(obs)
                    # 处理可能的动作形状不匹配问题
                    if action.shape != env.action_space.shape:
                        if action.ndim == 2 and action.shape[0] == 1:
                            action = action.squeeze(0)
                
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                
                if verbose and step_count % 100 == 0:
                    cool_power = round(np.mean(info.get("cool_power", 0)), 4)
                    heat_power = round(np.mean(info.get("heat_power", 0)), 4)
                    info_str = (
                        f"energy_cost: {round(info.get('energy_cost', 0), 4)}, "
                        f"target_cost: {round(info.get('target_cost', 0), 4)}, "
                        f"switch_cost: {round(info.get('switch_cost', 0), 4)}, "
                        f"cool_power: {cool_power}, heat_power: {heat_power}"
                    )
                    print(f"步骤 {step_count} | {info_str}")

            # 检查环境是否成功运行到指定步数
            if step_count >= max_steps and not terminated and not truncated:
                if verbose:
                    print(f"成功创建环境 #{len(valid_envs)+1} - 运行了 {step_count} 步")
                valid_envs.append(env)
                valid_config_paths.append(current_config_path)
            else:
                if verbose:
                    reason = "终止" if terminated else "截断" if truncated else f"仅运行 {step_count} 步"
                    print(f"环境验证失败 - 原因: {reason}")
                # 删除无效的配置文件
                try:
                    os.remove(current_config_path)
                except OSError:
                    pass
                
        except Exception as e:
            if verbose:
                print(f"环境创建/运行时出错: {str(e)}")
            # 删除可能已创建但无效的配置文件
            try:
                os.remove(current_config_path)
            except (OSError, UnboundLocalError):
                pass
            continue
    
    if not valid_envs:
        raise RuntimeError(f"无法创建有效的环境 - 在 {max_attempts} 次尝试后")
    
    if verbose:
        print(f"\n成功创建并验证了 {len(valid_envs)} 个环境")
        print("有效的配置文件路径:")
        for i, path in enumerate(valid_config_paths, 1):
            print(f"{i}. {path}")
    
    return valid_envs, valid_config_paths

if __name__ == "__main__":
    # 使用示例
    valid_envs, valid_configs = create_and_validate_hvac_env(
        pid_type="temperarure",
        task_config_path="./gen_env/hvac_task_0814_{}_time.pkl",  # 注意{}用于插入序号
        max_steps=1000,
        num_valid_envs=500,
        verbose=True
    )
    
    # 现在可以使用 valid_envs 列表中的环境和对应的配置文件路径
    print(f"\n获得 {len(valid_envs)} 个有效环境及其配置文件:")
    for env, config in zip(valid_envs, valid_configs):
        print(f"- 环境ID: {id(env)}, 配置文件: {config}")
