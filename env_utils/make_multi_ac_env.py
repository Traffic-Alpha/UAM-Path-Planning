from typing import List, Dict
from env_utils.ac_env import ACEnvironment
from env_utils.ac_wrapper import ACEnvWrapper
from env_utils.pz_env import ACEnvironmentPZ

from torchrl.envs import (
    ParallelEnv,
    TransformedEnv,
    RewardSum,
    VecNorm
)
from env_utils.torchrl_pz_wrapper import PettingZooWrapper # 对原始的 torchrl 的 wrapper 进行了修改


def make_multi_envs(
        ac_ids:List[str],
        sumo_cfg:str, net_file:str,
        num_seconds:int, use_gui:bool,
        action_space:Dict[str, int],
        log_file:str, device:str='cpu',
        **output_files
    ):
    ac_env = ACEnvironment(
        # **output_files,
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        num_seconds=num_seconds,
        ac_ids=ac_ids,
        # tls_action_type='choose_next_phase_syn',
        use_gui=use_gui,
    )
    ac_env = ACEnvWrapper(ac_env, filepath=log_file)
    ac_env = ACEnvironmentPZ(ac_env, action_space)
    ac_env = PettingZooWrapper(
        ac_env,
        group_map={'agents':ac_ids}, # agent 可以分类, 例如不同动作空间大小
        categorical_actions=False,
        use_mask=False, # 智能体数量动态变化, 手动将 obs 和 reward 设置为 0
        device=device,
        done_on_any=False # 所有都结束才结束
    )
    ac_env = TransformedEnv(ac_env)
    ac_env.append_transform(RewardSum(in_keys=[ac_env.reward_key]))
    ac_env.append_transform(VecNorm(in_keys=[ac_env.reward_key]))

    return ac_env

def make_parallel_env(
        num_envs:int,
        tls_ids:List[str],
        sumo_cfg:str, net_file:str,
        num_seconds:int, use_gui:bool,
        action_space:Dict[str, int],
        log_file:str,
        device:str='cpu'
    ):
    env = ParallelEnv(
        num_workers=num_envs,
        create_env_fn=make_multi_envs,
        create_env_kwargs=[{
            "tls_ids": tls_ids,
            "sumo_cfg": sumo_cfg,
            "num_seconds": num_seconds,
            "net_file": net_file,
            "action_space": action_space,
            "use_gui" : use_gui,
            "log_file": log_file+f'/{i}',
            "device": device,
        }
        for i in range(num_envs)]
    )

    return env