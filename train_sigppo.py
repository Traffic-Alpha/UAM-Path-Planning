'''
@Author: Ricca
@Date: 2024-07-16
@Description: 基于 Stabe Baseline3 控制单飞行汽车接送乘客
@LastEditTime:
'''
import argparse
import os
import torch
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from env_utils.make_tsc_env import make_env
from train_utils.sb3_utils import BestVecNormalizeCallback, linear_schedule,cosine_annealing_schedule

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
    EvalCallback
)

path_convert = get_abs_path(__file__)
logger.remove()
set_logger(path_convert('./'), file_log_level="ERROR", terminal_log_level="ERROR")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters.')
    parser.add_argument('--env_name', type=str, default="detroit_UAM", help='The name of environment, detroit_UAM, berlin_UAM')
    parser.add_argument('--passenger_len', type=int, default=5, help='The number of passengers')
    parser.add_argument('--passenger_type', type=str, default="real_time", help='fix or real time')
    parser.add_argument('--snir_min', type=int, default=-17, help='The threshold of SNIR') # 最小SNIR值，小于这个值的乘客不参与训练
    parser.add_argument('--num_envs', type=int, default=1, help='The number of environments')
    parser.add_argument('--n_steps', type=int, default=2000, help='The number of steps in each environment')
    parser.add_argument('--policy_model', type=str, default="fusion_models_v8", help='policy network: baseline_models or fusion_models_4')
    parser.add_argument('--features_dim', type=int, default=2048, help='The dimension of output features 64')
    parser.add_argument('--lr', type=float, default=5e-5, help='The learning rate of PPO')
    parser.add_argument('--batch_size', type=int, default=400, help='The batch size of PPO')
    parser.add_argument('--num_seconds', type=int, default=2500, help='exploration steps')
    parser.add_argument('--cuda_id', type=int, default=0, help='The id of cuda device')
    args = parser.parse_args()  # Parse the arguments
    device = f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu'

    # #########
    # Init
    # #########
    sumo_cfg = path_convert(f"./sumo_envs/{args.env_name}/{args.env_name}.sumocfg")
    net_file = path_convert(f"./sumo_envs/{args.env_name}/{args.env_name}.net.xml")
    snir_files = {
        '100': path_convert(f"./sumo_envs/{args.env_name}/{args.env_name}_SNIR_100.txt") # 高度： 文件路径
        # xxx, 这里可以添加不同高度的文件
    }

    if args.env_name == "berlin_UAM": # 3km, P4
        aircraft_inits = {
            'drone_1': {
                "aircraft_type": "drone",
                "action_type": "horizontal_multi_movement",
                "position": (2600, 1500, 100), "speed": 100, "heading": (1, 1, 0), "communication_range": 120,
                "if_sumo_visualization": True, "img_file": path_convert('./asset/drone.png'),
                # "custom_update_cover_radius":custom_update_cover_radius # 使用自定义覆盖范围的计算
            },
        }
        passenger_seq = {
            'num_seconds': args.num_seconds + 10,
            "passenger_seq": {
                "step_0": [[(2220, 1400), (1850, 1100)], [(1450, 2000), (1270, 450)], [(650, 1000), (1700, 620)]],
                "step_5": [[(1010, 2220), (400, 1200)]],
            }
        }

    else: # "berlin_UAM"，P5
        aircraft_inits = {
            'drone_1': {
                "aircraft_type": "drone",
                "action_type": "horizontal_multi_movement",
                "position": (6290, 9450, 100), "speed": 320, "heading": (1, 1, 0), "communication_range": 120,
                "if_sumo_visualization": False, "img_file": path_convert('./asset/drone.png'),
            },
        }

        passenger_seq = {
            'num_seconds': args.num_seconds + 10,
            "passenger_seq": {
                "step_0": [[(6594, 7563), (9570, 7210)], [(7494, 5855), (1375, 6530)]],
                "step_5": [[(3500, 5220), (5100, 2500)]],
                "step_8": [[(2175, 6906), (2234, 3850)]],
                "step_15": [[(3781, 3438), (7830, 2430)]],
            }
        }

    # #########
    # Save Path
    # #########
    # 不同乘客数量，不同SNIR_min，保存不同的log文件和model
    param_name = f'explore_{args.num_seconds}_n_steps_{args.n_steps}_lr_{str(args.lr)}_batch_size_{args.batch_size}'
    log_path = path_convert(f'./{args.passenger_type}/{args.env_name}/P{args.passenger_len}/sinr_{args.snir_min}/{args.policy_model}/{param_name}/logs/')
    model_path = path_convert(f'./{args.passenger_type}/{args.env_name}/P{args.passenger_len}/sinr_{args.snir_min}/{args.policy_model}/{param_name}/models/')
    tensorboard_path = path_convert(f'./{args.passenger_type}/{args.env_name}/P{args.passenger_len}/sinr_{args.snir_min}/{args.policy_model}/{param_name}/tensorboard/')

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    params = {
        'num_seconds':args.num_seconds,
        'sumo_cfg':sumo_cfg,
        'use_gui':False,
        "net_file": net_file,
        "snir_files": snir_files,
        'log_file':log_path,
        'aircraft_inits':aircraft_inits,
        'passenger_seq':passenger_seq,
        "snir_min":args.snir_min,
    }
    env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(args.num_envs)]) # multiprocess
    # env = VecNormalize(env, norm_obs=False, norm_reward=True)
    env = VecNormalize(env, norm_obs=True, norm_obs_keys=["ac_attr","passen_attr","passen_mask","sinr_attr","uncertainty_attr"], norm_reward=True)

    # #########
    # Callback
    # #########
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=model_path,
        save_vecnormalize=True
    )
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=500,
        verbose=True
    )  # 何时停止
    eval_callback = BestVecNormalizeCallback(
        env,
        best_model_save_path=model_path,
        callback_after_eval=stop_callback,  # 每次验证之后调用, 是否已经不变了, 需要停止
        eval_freq=500,  # 每次更新的样本数量为 n_steps*NUM_CPUS, n_steps 太小可能会收敛到局部最优
        verbose=1
    )  # 保存最优模型

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    # #########
    # Training
    # #########
    if args.policy_model.split("_")[0] == "baseline":
        from train_utils.baseline_models import CustomModel
        policy_models = CustomModel
    elif args.policy_model.split("_")[0] == "fusion":
        model_version = args.policy_model.split("_")[-1]
        # if model_version == "1":
        #     from train_utils.fusion_models_v1 import FusionModel
        # elif model_version == "2":
        #     from train_utils.fusion_models_v2 import FusionModel
        # elif model_version == "3":
        #     from train_utils.fusion_models_v3 import FusionModel # berlin_UAM
        # elif model_version == "4":
        #     from train_utils.fusion_models_v4 import FusionModel
        # elif model_version == "5":
        #     from train_utils.fusion_models_v5 import FusionModel
        # elif model_version == "6":
        #     from train_utils.fusion_models_v6 import FusionModel
        # elif model_version == "7":
        #     from train_utils.fusion_models_v7 import FusionModel
        # from train_utils.fusion_models_v7 import FusionModel  # before 6, no batch normalizaton
        from train_utils.fusion_models_v8 import FusionModel # add batch normalization
        policy_models = FusionModel
    else:
        raise ValueError("Invalid policy network type.")

    policy_kwargs = dict(
        features_extractor_class=policy_models,
        features_extractor_kwargs=dict(features_dim=args.features_dim,), # 27 44 43 64
    )
    model = PPO(
                "MultiInputPolicy", # "MultiInputPolicy""MlpPolicy"
                env, 
                batch_size=args.batch_size, #256
                n_steps=args.n_steps,
                n_epochs=5, # 每次间隔 n_epoch 去评估一次
                learning_rate=linear_schedule(args.lr),
                verbose=True, 
                policy_kwargs=policy_kwargs, 
                tensorboard_log=tensorboard_path, 
                device=device
            )
    model.learn(total_timesteps=1e6, tb_log_name='UAM', callback=callback_list) #3e5 1e6
    
    # #################
    # 保存 model 和 env
    # #################
    env.save(f'{model_path}/last_vec_normalize.pkl')
    model.save(f'{model_path}/last_rl_model.zip')
    print('训练结束, 达到最大步数.')

    env.close()