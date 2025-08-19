#!/usr/bin/env python



import argparse
import os
import sys
import numpy as np
import random
import torch
import multiprocessing
import pickle
import gc
import yaml
import time
from pathlib import Path
from datetime import datetime
import logging
# 添加缺失的环境导入和创建方法
# from xenoverse.anyhvacv2.anyhvac_env_vis import HVACEnvVisible
from xenoverse.anyhvacv2.anyhvac_env import HVACEnv
from xenoverse.anyhvacv2.anyhvac_sampler import HVACTaskSampler
from xenoverse.anyhvacv2.anyhvac_solver import HVACSolverGTPID
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from xenoverse.anyhvacv2.rl_trainer import HVACRLTrainer
# 定位到项目根目录（假设 generate_hvac_data.py 位于 .../xenoverse/anyhvacv2/）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))  # 上溯到 xenoverse-develop

# 优先加载本地代码
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# 移到模块级别的函数，避免pickle问题
def generate_single_epoch(args):
    """生成单个epoch的数据（用于多进程）"""
    (task_config_path, visual, use_time, use_heat, 
     policies_to_use, model_paths, seed, output_path, 
     epoch_id, max_steps, per_step_random, policy_weights, 
     ppo_aug_config, policy_switch_prob, policy_keep_prob) = args
    
    # 在子进程中导入，避免循环导入
    from xenoverse.anyhvacv2.multiagent_data_generator import MultiAgentDataGenerator
    
    # 在子进程中创建新的生成器实例
    generator = MultiAgentDataGenerator(
        task_config_path=task_config_path,
        visual=visual,
        use_time_in_observation=use_time,
        use_heat_in_observation=use_heat,
        policies_to_use=policies_to_use,
        model_paths=model_paths,
        seed=seed + epoch_id if seed else None,
        multiprocess_mode=True  # 标记为多进程模式
    )
    # 设置新的配置
    generator.per_step_random_policy = per_step_random
    generator.policy_weights = policy_weights
    generator.augment_config = ppo_aug_config
    generator.policy_switch_prob = policy_switch_prob
    generator.policy_keep_prob = policy_keep_prob
    
    # 生成数据
    data = generator.generate_multiagent_data(epoch_id, max_steps)
    generator.save_multiagent_data(data, output_path, epoch_id)
    
    # 清理内存
    del data
    del generator
    gc.collect()
    
    print(f"Process completed epoch {epoch_id}")
    return epoch_id

class OneClickDataGenerator:
    """一键数据生成器"""
    
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.setup_directories()
        self.setup_logging()
        
    def load_config(self, config_path):
        """加载配置"""
        default_config = {
            # 基础配置
            'mode': 'quick',  # quick, full, custom
            'seed': 42,
            'num_workers': 60,
            
            # 环境配置
            'env': {
                'task_config_path': './gen_env/hvac_task_time.pkl',
                'max_steps': 5040 , # 5040
                'use_time_in_observation': False,
                'use_heat_in_observation': True,
                'visual': False
            },
            
            # 训练配置
            'training': {
                'enabled': True,
                'algorithm': 'ppo',  # ppo, sac, recurrent_ppo
                'total_steps': 600000,  # quick模式
                'n_envs': 12,
                'vec_env_type': 'dummy',  # dummy, subproc
                'save_path': "./models/mixed_training_0814/hvac_task_time.zip",

                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'vec_env_args': {"start_method": "spawn"}
            },
            
            # 数据生成配置
            'data_generation': {
                'output_path': '/mnt/disk1/shaoshengbo/data/hvac/hvac_multiagent/0814_81',
                'epochs': 1,  # quick模式
                'start_index': 0,
                'max_steps_per_epoch': 5040,  # 5040
                'policies_to_use': ['ppo', 'pid', 'random', 'ppo_aug', 'constant'],
                'mask_all_tag_prob': 0.15,
                'mask_episode_tag_prob': 0.15,
            
                # 新增：每步随机策略配置
                'per_step_random_policy': True,  # 是否每步随机选择策略
                'policy_weights': {  # 各策略的选择权重
                    'random': 0.05,
                    'constant': 0.1,
                    'pid': 0.1,
                    'ppo': 0.75,
                    'ppo_aug': 0.0
                },
            
                # 新增：策略切换配置
                'policy_switch_prob': 0.008,  # 切换到其他策略的概率
                'policy_keep_prob': 0.992,    # 保持当前策略的概率
                
                # 新增：PPO增强配置
                'ppo_aug_config': {
                    'noise_scale': 0.1,
                    'action_perturb_scale': 0.1
                }
            },
            
            # 数据增强配置
            'augmentation': {
                'enabled': False,
                'num_augmentations': 2,
                'noise_level': 0.1,
                'action_perturb_prob': 0.2
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self.merge_config(default_config, user_config)
        
        return default_config
    
    def merge_config(self, default, user):
        """递归合并配置"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self.merge_config(default[key], value)
            else:
                default[key] = value
    
    def setup_directories(self):
        """创建必要的目录"""
        dirs = [
            './tasks',
            './models', 
            './data/hvac_multiagent',
            './logs'
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def setup_logging(self):
        """设置logging系统"""
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"{log_dir}/data_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 使用标准logging而不是维护文件句柄
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.log_filename = log_filename  # 保存文件名而不是文件句柄
        
    def log(self, message):
        """记录日志"""
        self.logger.info(message)
    
    def generate_or_load_task(self):
        """生成或加载HVAC任务配置"""
        task_path = self.config['env']['task_config_path']
        if os.path.exists(task_path):
            self.log(f"Loading existing task from {task_path}")
            with open(task_path, 'rb') as f:
                task = pickle.load(f)
        else:
            self.log("Sampling new HVAC tasks...")
            task = HVACTaskSampler(control_type='Temperature')
            with open(task_path, "wb") as f:
                pickle.dump(task, f)
            self.log(f"Saved new task config to {task_path}")
                
        self.print_task_summary(task)
        return task
    
    def print_task_summary(self, task):
        """打印任务摘要"""
        self.log("=" * 50)
        self.log("HVAC Task Configuration Summary:")
        self.log(f"  Building size: {task.get('n_width', 'N/A')} x {task.get('n_length', 'N/A')}")
        self.log(f"  Sensors: {task.get('n_sensors', 'N/A')}")
        self.log(f"  Heaters: {task.get('n_heaters', 'N/A')}")
        self.log(f"  Coolers: {task.get('n_coolers', 'N/A')}")
        self.log(f"  Ambient temp: {task.get('ambient_temp', 'N/A'):.1f}°C")
        self.log(f"  Target temp: {task.get('target_temperature', 'N/A')}")
        self.log("=" * 50)
    
    def check_existing_resources(self):
        """检查是否已有模型和任务文件"""
        model_path = self.config['training']['save_path']
        task_path = self.config['env']['task_config_path']
        
        model_exists = os.path.exists(model_path)
        task_exists = os.path.exists(task_path)
        
        self.log(f"Checking existing resources:")
        self.log(f"  Model file ({model_path}): {'Found' if model_exists else 'Not found'}")
        self.log(f"  Task file ({task_path}): {'Found' if task_exists else 'Not found'}")
        
        return model_exists, task_exists
    
    def train_rl_model(self, task, force_train=False):
        """训练RL模型"""
        model_path = self.config['training']['save_path']
        
        # 检查是否已有模型文件
        if os.path.exists(model_path) and not force_train:
            self.log(f"Model already exists at {model_path}, skipping training...")
            return model_path
        
        if not self.config['training']['enabled']:
            self.log("Training disabled in config, skipping...")
            return None
        
        self.log("Starting RL model training...")
        train_config = self.config['training']
        
        # Define make_env as a nested function
        def make_env():
            env = HVACEnv()
            env.set_task(task)
            return env    
        
        # 使用真正的训练器
        trainer = HVACRLTrainer(
            env_maker=make_env,
            n_envs=12,  # 并行环境数
            vec_env_type="subproc",  # 向量环境类型
            algorithm="ppo",  # 可选 "ppo" 或 "sac"
            stage_steps=100,  # 每5000步统计一次平均奖励
            vec_env_args={
                "start_method": "spawn"
            },
            verbose=0,
            device="cuda"  # 使用GPU加速
        )

        trainer.train(total_steps=train_config['total_steps'])
        trainer.save_model(train_config['save_path'])
        
        self.log(f"Model saved to {train_config['save_path']}")
        return train_config['save_path']
    
    def generate_multiagent_data(self, task, model_path):
        """生成多智能体数据"""
        self.log("Starting multi-agent data generation...")
        gen_config = self.config['data_generation']

        # 准备模型路径
        model_paths = {}
        if model_path and os.path.exists(model_path):
            model_paths['ppo'] = model_path
            self.log(f"Using existing PPO model: {model_path}")
        
        output_path = gen_config['output_path']
        epochs = gen_config['epochs']
        start_index = gen_config['start_index']
        max_steps = gen_config['max_steps_per_epoch']
        workers = self.config['num_workers']
        
        # 新增配置
        per_step_random = gen_config.get('per_step_random_policy', True)
        policy_weights = gen_config.get('policy_weights', {})
        ppo_aug_config = gen_config.get('ppo_aug_config', {})
        policy_switch_prob = gen_config.get('policy_switch_prob', 0.008)
        policy_keep_prob = gen_config.get('policy_keep_prob', 0.992)
        
        self.log(f"Generating {epochs} epochs with {workers} workers...")
        self.log(f"Per-step random policy: {per_step_random}")
        self.log(f"Policy switch probability: {policy_switch_prob}")
        self.log(f"Policy keep probability: {policy_keep_prob}")
        self.log(f"Policies to use: {gen_config['policies_to_use']}")
        
        if workers == 1:
            # 单进程生成
            from xenoverse.anyhvacv2.multiagent_data_generator import MultiAgentDataGenerator

            generator = MultiAgentDataGenerator(
                task_config_path=self.config['env']['task_config_path'],
                visual=False,
                use_time_in_observation=self.config['env']['use_time_in_observation'],
                use_heat_in_observation=self.config['env']['use_heat_in_observation'],
                policies_to_use=gen_config['policies_to_use'],
                model_paths=model_paths,
                seed=self.config['seed']
            )
            
            # 设置新配置
            generator.per_step_random_policy = per_step_random
            generator.policy_weights = policy_weights
            generator.augment_config = ppo_aug_config
            generator.policy_switch_prob = policy_switch_prob
            generator.policy_keep_prob = policy_keep_prob
            
            self.log(f"Data generator initialized:")
            self.log(f"  Sensors: {generator.n_sensors}")
            self.log(f"  Coolers: {generator.n_coolers}")
            self.log(f"  Heaters: {generator.n_heaters}")
            self.log(f"  Policies: {generator.policies_to_use}")
            
            for epoch_id in range(start_index, start_index + epochs):
                data = generator.generate_multiagent_data(epoch_id, max_steps)
                generator.save_multiagent_data(data, output_path, epoch_id)
                self.log(f"Generated epoch {epoch_id - start_index + 1}/{epochs}")
                del data
                gc.collect()
        else:
            # 多进程生成 - 使用新的方法
            self.log("Using multiprocess generation...")
            
            # 准备参数列表
            args_list = []
            base_seed = self.config['seed'] if self.config['seed'] is not None else 0
            # for i, epoch_id in enumerate(range(start_index, start_index + epochs)):
            for epoch_id in range(start_index, start_index + epochs):
                current_seed = base_seed + random.randint(0, 100000000) # 加上一个大随机数作为偏移，确保每次运行的随机性
                args = (
                    self.config['env']['task_config_path'],
                    False,  # visual必须为False
                    self.config['env']['use_time_in_observation'],
                    self.config['env']['use_heat_in_observation'],
                    gen_config['policies_to_use'],
                    model_paths,
                    current_seed,
                    output_path,
                    epoch_id,
                    max_steps,
                    per_step_random,  
                    policy_weights,   
                    ppo_aug_config,   
                    policy_switch_prob,  
                    policy_keep_prob    
                )
                args_list.append(args)
            
            # 设置multiprocessing启动方法
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # 已经设置过了
            
            # 使用进程池执行
            with multiprocessing.Pool(workers) as pool:
                results = pool.map(generate_single_epoch, args_list)
            
            self.log(f"Completed epochs: {results}")
        
        self.log(f"Data generation completed! Data saved to {output_path}")
        return

    def simulate_data_generation(self, gen_config):
        """模拟数据生成过程"""
        epochs = gen_config['epochs']
        output_path = Path(gen_config['output_path'])
        
        for epoch in range(epochs):
            epoch_path = output_path / f'record-{epoch:06d}'
            os.makedirs(epoch_path, exist_ok=True)
            
            # 模拟生成数据文件
            n_agents = 5  # 假设5个传感器/智能体
            n_timesteps = gen_config['max_steps_per_epoch']
            
            # 生成模拟数据
            for agent_id in range(n_agents):
                # 观测数据
                obs = np.random.randn(n_timesteps, 3).astype(np.float32)
                np.save(epoch_path / f'observations_{agent_id}.npy', obs)
                
                # 行为动作
                actions_b = np.random.randn(n_timesteps, 2).astype(np.float32)
                np.save(epoch_path / f'actions_behavior_{agent_id}.npy', actions_b)
                
                # 标签动作
                actions_l = np.random.randn(n_timesteps, 2).astype(np.float32)
                np.save(epoch_path / f'actions_label_{agent_id}.npy', actions_l)
                
                # 标签
                tags = np.random.randint(0, 4, size=n_timesteps).astype(np.int32)
                np.save(epoch_path / f'tags_{agent_id}.npy', tags)
            
            # 全局数据
            rewards = np.random.randn(n_timesteps).astype(np.float32)
            resets = np.random.choice([0, 1], size=n_timesteps, p=[0.98, 0.02]).astype(bool)
            
            np.save(epoch_path / 'rewards.npy', rewards)
            np.save(epoch_path / 'resets.npy', resets)
            np.save(epoch_path / 'obs_graph.npy', np.eye(n_agents, dtype=np.float32))
            np.save(epoch_path / 'agent_graph.npy', np.ones((n_agents, n_agents), dtype=np.float32))
            
            self.log(f"Generated epoch {epoch + 1}/{epochs}")
    
    def run_augmentation(self):
        """运行数据增强"""
        if not self.config['augmentation']['enabled']:
            self.log("Data augmentation disabled, skipping...")
            return
        
        self.log("Starting data augmentation...")


        
        self.log("Data augmentation completed!")
    
    def print_summary(self):
        """打印总结"""
        self.log("\n" + "=" * 50)
        self.log("Data Generation Summary:")
        self.log(f"  Mode: {self.config['mode']}")
        self.log(f"  Total epochs generated: {self.config['data_generation']['epochs']}")
        self.log(f"  Data path: {self.config['data_generation']['output_path']}")
        
        if self.config['training']['enabled']:
            self.log(f"  Model trained: Yes ({self.config['training']['algorithm'].upper()})")
            self.log(f"  Training steps: {self.config['training']['total_steps']}")
        else:
            self.log(f"  Model trained: No")
        
        if self.config['augmentation']['enabled']:
            self.log(f"  Data augmented: Yes (x{self.config['augmentation']['num_augmentations']})")
        else:
            self.log(f"  Data augmented: No")
        
        self.log("=" * 50)
    
    def run(self):
        """运行完整的数据生成流程"""
        start_time = time.time()
        
        try:
            # 0. 检查已有资源
            self.log("Step 0: Checking existing resources")
            model_exists, task_exists = self.check_existing_resources()
            
            # 1. 生成或加载任务
            self.log("\nStep 1: Task Configuration")
            task = self.generate_or_load_task()
            
            # 2. 训练RL模型（如果需要）
            self.log("\nStep 2: RL Model Training")
            if model_exists and task_exists:
                self.log("Both model and task files exist, skipping training...")
                model_path = self.config['training']['save_path']
            else:
                model_path = self.train_rl_model(task)
            # 3. 生成多智能体数据
            self.log("\nStep 3: Multi-Agent Data Generation")
            self.generate_multiagent_data(task, model_path)
            
            # 4. 数据增强（如果启用）
            self.log("\nStep 4: Data Augmentation")
            self.run_augmentation()
            
            # 5. 打印总结
            self.print_summary()
            
            elapsed_time = time.time() - start_time
            self.log(f"\nTotal execution time: {elapsed_time:.2f} seconds")
            self.log("Data generation completed successfully!")
            
        except Exception as e:
            self.log(f"Error during data generation: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            raise


def create_preset_configs():
    """创建预设配置"""
    configs = {
        'quick': {
            'mode': 'quick',
            'training': {
                'enabled': True,
                'total_steps': 600000
            },
            'data_generation': {
                'epochs': 1,
                'per_step_random_policy': True,
                'policies_to_use': ['random', 'pid', 'ppo', 'ppo_aug', 'constant'],
                'policy_weights': {
                    'random': 0.05,
                    'constant': 0.1,
                    'pid': 0.1,
                    'ppo': 0.75,
                    'ppo_aug': 0.0
                },
                'policy_switch_prob': 0.008,
                'policy_keep_prob': 0.992
            }
        },
    }
    
    return configs


def main():
    parser = argparse.ArgumentParser(description='一键生成HVAC多智能体数据')
    parser.add_argument('--mode', type=str, default='quick', 
                       choices=['quick', ],
                       help='运行模式')
    parser.add_argument('--config', type=str, default=None,
                       help='自定义配置文件路径（YAML格式）')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子')
    parser.add_argument('--workers', type=int, default=None,
                       help='并行工作进程数')
    parser.add_argument('--epochs', type=int, default=None,
                       help='生成的epoch数量')
    parser.add_argument('--training_steps', type=int, default=None,
                       help='训练步数')
    parser.add_argument('--force_train', default=False,
                       help='强制重新训练，即使已有模型文件')
    parser.add_argument('--policy_switch_prob', type=float, default=None,
                       help='策略切换概率')
    parser.add_argument('--policy_keep_prob', type=float, default=None,
                       help='策略保持概率')
    
    args = parser.parse_args()
    
    # 根据模式选择配置
    if args.mode == 'custom':
        if not args.config:
            print("Error: Custom mode requires --config parameter")
            sys.exit(1)
        generator = OneClickDataGenerator(args.config)
    else:
        # 使用预设配置
        preset_configs = create_preset_configs()
        config = preset_configs[args.mode]
        
        # 覆盖命令行参数
        if args.seed is not None:
            config['seed'] = args.seed
        if args.workers is not None:
            config['num_workers'] = args.workers
        if args.epochs is not None:
            config['data_generation']['epochs'] = args.epochs
        if args.training_steps is not None:                                                                                                                                                                                                                                                                                                                                                                                                       
            config['training']['total_steps'] = args.training_steps
        if args.policy_switch_prob is not None:
            config['data_generation']['policy_switch_prob'] = args.policy_switch_prob
        if args.policy_keep_prob is not None:
            config['data_generation']['policy_keep_prob'] = args.policy_keep_prob
        
        # 创建临时配置文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_config_path = f.name
        
        generator = OneClickDataGenerator(temp_config_path)
        
        # 清理临时文件
        os.unlink(temp_config_path)
    
    # 运行数据生成
    generator.run()


if __name__ == "__main__":
    main()
