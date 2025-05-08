'''
@Author: WANG Maonan
@Date: 2023-09-14 13:47:34
@Description: Check aircraft and vehicle ENV
+ Two types of aircraft, custom image
@LastEditTime: 2023-09-25 14:20:32
'''
import math
import numpy as np
from loguru import logger
from typing import List
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.format_dict import dict_to_str
# from env_utils.make_tsc_env import make_env
from env_utils.aircraft_snir_env import ACSNIREnvironment
from env_utils.ac_wrapper import ACEnvWrapper
from env_utils.vis_snir import render_map


path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))

def custom_update_cover_radius(position:List[float], communication_range:float) -> float:
    """自定义的更新地面覆盖半径的方法, 在这里实现您的自定义逻辑

    Args:
        position (List[float]): 飞行器的坐标, (x,y,z)
        communication_range (float): 飞行器的通行范围
    """
    height = position[2]
    cover_radius = height / np.tan(math.radians(75/2))
    return cover_radius


def make_env(
        num_seconds:int,sumo_cfg:str,use_gui:bool,
        net_file:str, snir_files:dict,
        log_file:str, aircraft_inits:dict,
        passenger_seq:list,snir_min:int,
        ):

    ac_env = ACSNIREnvironment(
        sumo_cfg=sumo_cfg,
        num_seconds=num_seconds,
        net_file=net_file,
        radio_map_files=snir_files,
        aircraft_inits=aircraft_inits,
        use_gui=use_gui
    )
    ac_wrapper = ACEnvWrapper(env=ac_env, aircraft_inits=aircraft_inits, passenger_info=passenger_seq, snir_min=snir_min)
    # ac_env = Monitor(ac_wrapper, filename=f'{log_file}/{env_index}')
    return ac_wrapper

if __name__ == '__main__':
    # env_name = "berlin_UAM"
    env_name = "detroit_UAM"
    sumo_cfg = path_convert(f"./sumo_envs/{env_name}/{env_name}.sumocfg")
    net_file = path_convert(f"./sumo_envs/{env_name}/{env_name}.net.xml")
    snir_files = {
        '100': path_convert(f"./sumo_envs/{env_name}/{env_name}_SNIR_100.txt"),
        # xxx, 这里可以添加不同高度的文件
    }

    aircraft_inits = {
        'drone_1': {
            "aircraft_type": "drone",
            "action_type": "horizontal_multi_movement",
            "position":(6290, 9450, 100), "speed": 270, "heading":(1,1,0), "communication_range":60,
            "if_sumo_visualization": True, "img_file": path_convert('./asset/drone.png'),
            # "custom_update_cover_radius":custom_update_cover_radius # 使用自定义覆盖范围的计算
        },

    }

    passenger_seq = {
        'num_seconds': 100 + 10,
        "passenger_seq": {  # (9570, 7000)
            # "step_0": [[(2220, 1400), (2000, 1100)], [(1400, 1680), (1270, 450)], [(650, 1000), (1700, 620)]], # models_4
            "step_0": [[(6594, 7563), (9570, 7210)], [(7494, 5855), (1375, 6530)]],
            "step_5": [[(3500, 5220), (5100, 2500)]],
            "step_8": [[(2175, 6906), (2234, 3850)]],
            "step_15": [[(3781, 3438), (7830, 2430)]],
        }
    }

    env = make_env(
        sumo_cfg=sumo_cfg,
        num_seconds=100,
        net_file=net_file,
        snir_files=snir_files,
        aircraft_inits=aircraft_inits,
        passenger_seq=passenger_seq,
        snir_min=-17,
        log_file="./check_log",
        use_gui=False
    )

    # Check Env
    print(env.action_space.n)

    done = False
    env.reset()
    # import time; time.sleep(3)
    while not done:
        action = {
            "drone_1": 2, # np.random.randint(9),
        }
        states, rewards, truncated, done, infos = env.step(actions=action)
        if done:
            print("done")
        logger.info(f'SIM: State: \n{dict_to_str(states)} \nReward:\n {rewards}')
    print(env.ac_history_pos)
    # 进行可视化
    render_map(
        x_min=env.x_min,
        y_min=env.y_min,
        x_max=env.x_max,
        y_max=env.y_max,
        resolution=env.resolution,
        grid_z=env.grid_z,
        trajectories=env.ac_history_pos,
        goal_points=env.goal_seq,
        speed=320,  # 60: 50*50, 100: 30*30
        snir_threshold=-17,
        img_path=path_convert("./check_log/snir.jpg")
    )
    env.close()

