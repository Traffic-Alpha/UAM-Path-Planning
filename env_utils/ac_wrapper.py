'''
@Author: Ricca
@Date: 2024-07-16
@Description: 处理 ACEnvironment
+ state wrapper: 获得每个 aircraft 在覆盖范围内车辆的信息, 只有 aircraft与车辆进行通信
+ reward wrapper: aircraft 覆盖车辆个数
@LastEditTime:
'''
import numpy as np
import gymnasium as gym
from collections import deque

import torch
from gymnasium.core import Env
from typing import Any, SupportsFloat, Tuple, Dict, List
from collections import defaultdict
import math

import logging
# debug_log_file = './debug.log'
# logging.basicConfig(filename=debug_log_file, filemode='w', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class ACEnvWrapper(gym.Wrapper):
    """Aircraft Env Wrapper for single junction with tls_id
    """

    def __init__(self, env: Env, aircraft_inits, passenger_info: Dict, snir_min: int,max_states: int = 3) -> None:
        super().__init__(env)
        self.max_states = max_states
        self.initial_points = {
            ac_id: ac_value["position"]
            for ac_id, ac_value in self.env.tsc_env.aircraft_inits.items()
        }
        self._pos_set = deque([self._get_initial_state()] * max_states, maxlen=max_states)  # max state : 3
        self.speed = aircraft_inits["drone_1"]["speed"]
        self.agent_num = len(aircraft_inits.keys())
        # reset 的时候会获得的静态信息
        self.x_min, self.y_min, self.x_max, self.y_max, self.resolution = None, None, None, None, None
        self.grid_z = None
        self.scaled_snir_grid = None
        self.ac_history_pos = None  # 记录 aircraft 的历史轨迹, 用于绘图, 需要在 reset 的时候哦初始化
        self.actru_trajectory = None  # 记录ac的最后一个点，防止reset后，记录被清除，用于绘图
        self.ac_sit_flag = None  # 飞机是否有剩余座位
        self.uncertainty_matrix = None  # 未来预测的不确定性矩阵
        self.side_length = None  # 地图边长
        self.uncertainty_x_max, self.uncertainty_y_max = 33, 33  # 不确定性矩阵的大小

        # SNIR threshold
        self.snir_min = snir_min

        # passenger attributes 初始化用户信息
        self.total_step = passenger_info["num_seconds"]  # 总步数
        self.passenger_step_dict = passenger_info["passenger_seq"]  # 对应时间的乘客列表字典
        self.passenger_step_num = {}  # 对应时间的乘客数量字典
        self.goal_seq = []
        max_passen_num = 0
        for key, _seq in self.passenger_step_dict.items():
            max_passen_num += len(_seq)
            self.goal_seq += self.passenger_step_dict[key]
            self.passenger_step_num[int(key.split("_")[-1])] = max_passen_num

        self.max_passen_num = int(max_passen_num)  # 最大乘客数
        self.passen_mask = np.zeros((self.total_step, self.max_passen_num))  # 创建mask
        # passen_step_num_array = self.passenger_step_num[0]*np.zeros((self.total_step,1))
        self.passen_mask = np.ones((self.total_step, self.max_passen_num))  # 创建mask

        for i in range(self.total_step):
            if i in self.passenger_step_num.keys():
                passen_len = self.passenger_step_num[i]
                self.passen_mask[i:, :passen_len] = 0

        # 乘客信息矩阵，第1，2列为乘客起点相对飞行器的坐标，
        # 第3，4列为乘客终点相对飞行器的坐标，第5列为乘客与飞行器的距离，第6列为乘客与起点的距离，
        # 第7列为乘客与终点的距离，第8列为乘客是否在飞行器上，第9列为乘客是否被服务过
        self.padded_passen_attr = np.zeros((self.max_passen_num, 9))

        # action 转换
        speed = self.speed

        self.air_actions = {
            0: (speed, 0), # -> 右
            1: (speed, 1), # ↗ 右上
            2: (speed, 2), # ↗ 右上
            3: (speed, 3), # ↗ 右上
            4: (speed, 4), # ↑ 正上
            5: (speed, 5), # ↖ 左上
            6: (speed, 6), # ↖ 左上
            7: (speed, 7), # ↖ 左上
            8: (speed, 8), # ← 左
            9: (speed, 9), # ↙ 左下
            10: (speed, 10),# ↙ 左下
            11: (speed, 11),# ↙ 左下
            12: (speed, 12), # ↓ 正下
            13: (speed, 13), # ↘ 右下
            14: (speed, 14), # ↘ 右下
            15: (speed, 15), # ↘ 右下
            # ...
        }
        self.opsite_actions = {
            0:(speed, 8),
            1:(speed, 9),
            2:(speed, 10),
            3:(speed, 11),
            4:(speed, 12),
            5:(speed, 13),
            6:(speed, 14),
            7:(speed, 15),
            8:(speed, 0),
            9:(speed, 1),
            10:(speed, 2),
            11:(speed, 3),
            12:(speed, 4),
            13:(speed, 5),
            14:(speed, 6),
            15:(speed, 7),
        }

        self.re_step_count = 0 # 记录回退次数
        self.last_feature_set = None # 记录上一次的feature set

    def _get_initial_state(self) -> List[int]:
        return [0, 0, 2]  # x,y,seat_num

    def get_state_set(self):
        return np.array(self.state_set, dtype=np.float32)

    def get_distance(self, x: List, y: List) -> float:
        dist = np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
        return round(dist)

    def arrive_goal(self, x: List[int], y: List[int]):
        if self.get_distance(x, y) <= self.speed:
            return True
        else:
            return False

    def all_passenger_arrive(self):
        return np.all(np.array(self.passen_flag_list) == 1)

    def get_relative_pos(self, aircraft_id, pos) -> List:
        _init_points = self.initial_points[aircraft_id]
        pos_new = [pos[0] - _init_points[0], pos[1] - _init_points[1]]
        return pos_new

    def get_scaled_snir(self):  # 获取每个大格的均值，形成新的snir矩阵
        # x_max, y_max = self.grid_z.shape # 300*300
        _x_max, _y_max = int(self.side_length // self.resolution), int(
            self.side_length // self.resolution)  # 300*300 每个小格的边长是resolution，长和宽分别有多少格小格
        scale_grid_num = int(self.speed / self.resolution)  # 一个大格里包含多少个小格，大格的边长是速度 速度需要是resolution的倍数
        scaled_x_max, scaled_y_max = int(math.ceil(self.side_length / self.speed)), int(
            math.ceil(self.side_length / self.speed))  # 长和宽总分别有多少大格
        new_grid_z = np.zeros((scaled_x_max, scaled_y_max))  # (30*30)
        # 生成大格的snir矩阵
        for x in range(0, _x_max, scale_grid_num):
            for y in range(0, _y_max, scale_grid_num):
                snir = []
                for k in range(x, x + scale_grid_num):
                    for l in range(y, y + scale_grid_num):
                        if k < _x_max and l < _y_max:
                            snir.append(self.grid_z[k, l])
                mean_snir = np.nanmean(snir)
                new_grid_z[int(x // scale_grid_num), int(y // scale_grid_num)] = mean_snir
        return new_grid_z

    def get_around_snir(self, x, y):
        _x_max, _y_max = self.scaled_snir_grid.shape
        sinr_now = self.scaled_snir_grid[x, y]
        around_snir_matrix = []
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if 0 <= i < _x_max and 0 <= j < _y_max:
                    around_snir_matrix.append(self.scaled_snir_grid[i, j])
                else:
                    around_snir_matrix.append(sinr_now)
        return np.array(around_snir_matrix).reshape(3, 3)

    def get_around_uncertainty(self, x, y):
        _x_max, _y_max = self.uncertainty_matrix.shape
        uncertainty_now = self.uncertainty_matrix[x, y]
        around_uncertainty_matrix = []
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if 0 <= i < _x_max and 0 <= j < _y_max:
                    around_uncertainty_matrix.append(self.uncertainty_matrix[i, j])
                else:
                    around_uncertainty_matrix.append(uncertainty_now)
        return np.array(around_uncertainty_matrix).reshape(3, 3)

    @property
    def action_space(self):
        return gym.spaces.Discrete(16)

    @property
    def observation_space(self):
        spaces = {
            "ac_attr": gym.spaces.Box(low=np.zeros((1, 9)), high=np.ones((1, 9)), shape=(1, 9)),
            "passen_attr": gym.spaces.Box(low=np.zeros((self.max_passen_num, 9)),
                                          high=np.ones((self.max_passen_num, 9)),
                                          shape=(self.max_passen_num, 9)),
            "passen_mask": gym.spaces.Box(low=0, high=1, shape=(self.max_passen_num,), dtype=np.int32),
            "sinr_attr": gym.spaces.Box(low=np.zeros((3, 3)), high=np.ones((3, 3)), shape=(3, 3)),
            "uncertainty_attr": gym.spaces.Box(low=np.zeros((3, 3)),
                                               high=np.ones((3, 3)),
                                               shape=(3, 3)),
        }
        dict_space = gym.spaces.Dict(spaces)
        return dict_space

    # Wrapper
    def state_wrapper(self, state, actions, infos):
        """自定义 state 的处理, 只找出与 aircraft 通信范围内的 vehicle
        """
        tree = lambda: defaultdict(tree)
        new_state = tree()

        aircraft = state['aircraft']
        grid = state['grid']["100"]
        _step = int(infos['step_time']) - self.re_step_count

        for ac_idx, (aircraft_id, aircraft_info) in enumerate(aircraft.items()):
            aircraft_pos = aircraft_info["position"]
            aircraft_x, aircraft_y, aircraft_h = round(aircraft_pos[0]), round(aircraft_pos[1]), round(aircraft_pos[2]) # 四舍五入取整
            aircraft_pos_new = self.get_relative_pos(aircraft_id, aircraft_pos)

            # Map attributes: snir
            scaled_grid_len = int(self.speed)
            scaled_x_max, scaled_y_max = self.scaled_snir_grid.shape
            x_scale, y_scale = aircraft_x // scaled_grid_len, aircraft_y // scaled_grid_len
            if x_scale >= scaled_x_max:
                x_scale = scaled_x_max - 1
            if x_scale < 0:
                x_scale = 0
            if y_scale >= scaled_y_max:
                y_scale = scaled_y_max - 1
            if y_scale < 0:
                y_scale = 0

            # 地理坐标系 (m, n) -> 矩阵坐标 (n, m)
            # snir_value = self.scaled_snir_grid[int(y_scale), int(x_scale)]
            around_snir_matrix = self.get_around_snir(int(y_scale), int(x_scale))

            # AC attributes:
            self.ac_history_pos[aircraft_id].append(aircraft_pos[:2])  # 维护ac的轨迹
            ac_state = aircraft_pos_new + [self.ac_seat_flag]
            self._pos_set.append(ac_state)  # ac时序

            new_state[aircraft_id]["pos_set"] = np.array(self._pos_set).reshape(1, -1)

            # uncertainty matrix 地理坐标系 (m, n) -> 矩阵坐标 (n, m)
            self.uncertainty_matrix[y_scale, x_scale] += 1
            around_uncertainty_matrix = self.get_around_uncertainty(int(y_scale), int(x_scale))

            # Passenger attributes
            passen_mask = self.passen_mask[_step].copy()
            passen_num = np.sum(passen_mask == 0)

            for i in range(int(passen_num)):
                on_ac_flag = self.padded_passen_attr[i, -2]  # 是否在飞行器上
                is_served_flag = self.padded_passen_attr[i, -1]  # 是否被服务过
                if not on_ac_flag and not is_served_flag:  # 乘客在起点
                    self.padded_passen_attr[i, 4] = self.get_distance(self.goal_seq[i][0], aircraft_pos)  # 乘客与飞行器的距离
                    self.padded_passen_attr[i, 5] = 0  # 乘客与起点的距离
                    self.padded_passen_attr[i, 6] = self.get_distance(self.goal_seq[i][0],
                                                                      self.goal_seq[i][1])  # 乘客与终点的距离
                elif on_ac_flag and not is_served_flag:  # 乘客在飞机上
                    self.padded_passen_attr[i, 4] = 0
                    self.padded_passen_attr[i, 5] = self.get_distance(aircraft_pos, self.goal_seq[i][0])
                    self.padded_passen_attr[i, 6] = self.get_distance(aircraft_pos, self.goal_seq[i][1])
                else:  # 乘客在终点 is_served_flag == 1
                    self.padded_passen_attr[i, 4] = self.get_distance(self.goal_seq[i][1], aircraft_pos)
                    self.padded_passen_attr[i, 5] = self.get_distance(self.goal_seq[i][1], self.goal_seq[i][0])
                    self.padded_passen_attr[i, 6] = 0

            passen_attr = self.padded_passen_attr.copy()
            passen_attr[passen_mask == 1] = 0
            ac_attr = np.concatenate(
                (
                    new_state[aircraft_id]["pos_set"],
                ), axis=1
            )

            feature_set = {
                "ac_attr": ac_attr,
                "passen_attr": passen_attr,
                "passen_mask": passen_mask.reshape(-1),
                "sinr_attr": around_snir_matrix.copy(),
                "uncertainty_attr": around_uncertainty_matrix.copy(),
            }

        return feature_set

    def reward_wrapper(self, states, infos, dones, actions) -> Tuple[float, bool, dict or None]:
        """自定义 reward 的计算
        """
        reward = 0
        aircraft = states['aircraft']
        grid = states['grid']["100"]
        _step = int(infos['step_time'])
        # passen_attr = self.padded_passen_attr[_step]
        passen_mask = self.passen_mask[_step]
        passen_num = np.sum(passen_mask == 0)
        updated_action = None

        for ac_idx, (aircraft_id, aircraft_info) in enumerate(aircraft.items()):
            aircraft_pos = aircraft_info["position"]
            action = actions[aircraft_id][-1]
            _x, _y, _h = round(aircraft_pos[0]), round(aircraft_pos[1]), round(aircraft_pos[2])

            scaled_grid_len = int(self.speed)
            grid_x_max, grid_y_max = self.scaled_snir_grid.shape
            x_scale, y_scale = int(_x // scaled_grid_len), int(_y // scaled_grid_len)
            if x_scale >= grid_x_max:
                x_scale = grid_x_max - 1
            if x_scale < 0:
                x_scale = 0
            if y_scale >= grid_y_max:
                y_scale = grid_y_max - 1
            if y_scale < 0:
                y_scale = 0

            step_reward = -2  # 每步惩罚
            reward += step_reward

            # 记录飞行过的格子, 地理坐标系 (m, n) -> 矩阵坐标 (n, m)
            if self.uncertainty_matrix[y_scale, x_scale] - 1 == 0: # 减掉自己到达的这一次
                new_area_reward = 1
            else: # 重复区域的惩罚
                new_area_reward = - 0.1 * (self.uncertainty_matrix[y_scale, x_scale] - 1)
            reward += new_area_reward

            # 如果一局游戏总是在某个格子来回超过100次，则停止
            if np.any(self.uncertainty_matrix > 100):
                dones = True
                self.actru_trajectory = self.ac_history_pos
                return reward, dones, updated_action

            # 惩罚ac飞出边界
            if _x < 0 or _x > self.x_max:
                bound_reward = -50
                reward += bound_reward
                return reward, dones, self.opsite_actions[action]

            if _y < 0 or _y > self.y_max:
                bound_reward = -50
                reward += bound_reward
                return reward, dones, self.opsite_actions[action]

            # 惩罚ac飞到snir较低的区域
            # 地理坐标系 (m, n) -> 矩阵坐标 (n, m)
            snir_value = self.scaled_snir_grid[y_scale, x_scale]
            if snir_value < self.snir_min:
                sinr_reward = -50
                reward += sinr_reward
                return reward, dones, self.opsite_actions[action]

            # 到达乘客目标点，且完成一次接送 +50
            # 乘客在等车时： on_ac_flag = 0,
            #             is_served_flag = 0,
            # 乘客在飞机上时： on_ac_flag = 1,
            #               is_served_flag = 0,
            # 乘客到达终点时： on_ac_flag = 0,
            #               is_served_flag = 1,
            for passen_idx, goal_points in enumerate(self.goal_seq[:int(passen_num)]):
                on_ac_flag = self.padded_passen_attr[passen_idx, -2]
                is_served_flag = self.padded_passen_attr[passen_idx, -1]
                # ac到达乘客起点，该乘客没有在飞机上，该乘客没有结束服务, 且ac有空位
                if self.arrive_goal(aircraft_pos,
                                    goal_points[0]) and not on_ac_flag and not is_served_flag and self.ac_seat_flag > 0:
                    start_reward = 200
                    reward += start_reward
                    on_ac_flag = 1
                    self.ac_seat_flag -= 1  # ac座位数量-1
                    self.ac_history_pos[aircraft_id].append(goal_points[0])

                if self.arrive_goal(aircraft_pos, goal_points[1]) and on_ac_flag:  # 到达乘客终点，且ac正载着该乘客
                    self.ac_history_pos[aircraft_id].append(goal_points[1])
                    end_reward = 300
                    reward += end_reward
                    is_served_flag = 1
                    on_ac_flag = 0
                    self.ac_seat_flag += 1  # ac释放一个空位
                self.padded_passen_attr[passen_idx, -2] = on_ac_flag
                self.padded_passen_attr[passen_idx, -1] = is_served_flag

            # 如果所有乘客已服务完成，游戏结束
            all_passen_serve = self.padded_passen_attr[:, -1]
            if np.all(all_passen_serve == 1):
                dones = True
                self.actru_trajectory = self.ac_history_pos
                return reward, dones, updated_action
        if dones:
            self.actru_trajectory = self.ac_history_pos
        return reward, dones, updated_action

    def reset(self, seed=1) -> Tuple[Any, Dict[str, Any]]:
        """reset 时初始化 (1) 静态信息; (2) 动态信息
        """
        state = self.env.reset()

        self.side_length = min(state['grid']['100'].x_max - state['grid']['100'].x_min, state['grid']['100'].y_max - state['grid']['100'].y_min)
        self.x_max = self.side_length  # side_length = min(x_max - x_min, y_max - y_min)
        self.y_max = self.side_length # 11130 self.side_length
        self.re_step_count = 0

        self.grid_z = state['grid']['100'].grid_z
        self.grid_z[np.isnan(self.grid_z)] = np.nanmin(self.grid_z)
        self.resolution = state['grid']['100'].resolution
        self.scaled_snir_grid = self.get_scaled_snir()
        self.ac_history_pos = {
            ac_id: list()
            for ac_id in self.env.tsc_env.aircraft_inits.keys()
        }  # 初始化 aircraft 的历史轨迹, 用于绘图

        self.ac_seat_flag = 2  # 初始化飞机座位数量

        # 初始化乘客信息
        # 乘客信息矩阵，第1，2列为乘客起点相对飞行器的坐标，第3，4列为乘客终点相对飞行器的坐标，第5列为乘客与飞行器的距离，第6列为乘客与起点的距离，第7列为乘客与终点的距离，第8列为乘客是否在飞行器上，第9列为乘客是否被服务过
        for s in range(self.max_passen_num):
            self.padded_passen_attr[s, :2] = self.get_relative_pos("drone_1", self.goal_seq[s][0])  # 初始化乘客的起点
            self.padded_passen_attr[s, 2:4] = self.get_relative_pos("drone_1", self.goal_seq[s][1])  # 初始化乘客的终点
            self.padded_passen_attr[s, -2] = 0  # 初始化用户是否在飞行器上：0不在，1在
            self.padded_passen_attr[s, -1] = 0  # 初始化用户被服务的状态：0未被服务，1已被服务

        # 初始化uncertainty matrix 以飞机一步飞行的格子数为一个大格，建立matrix
        # _x_length, _y_length = self.grid_z.shape
        _x_length, _y_length = self.x_max // self.resolution, self.y_max // self.resolution
        grid_length = int(self.speed / self.resolution)  # 一个大格包含的格子数
        self.uncertainty_x_max = math.ceil(_x_length / grid_length)  # 计算x方向大格子数
        self.uncertainty_y_max = math.ceil(_y_length / grid_length)  # 计算y方向大格子数
        self.uncertainty_matrix = np.zeros((self.uncertainty_x_max, self.uncertainty_y_max))

        state = self.state_wrapper(state=state, actions={"drone_1": (0, 0)}, infos={"step_time": 0})

        return state, {'step_time': 0}

    def step(self, actions: Dict[str, int]) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        # {"d1":1, "d2": 2} --> {"d1":(5,1), "d2":(5,2)}
        # self.step += 1
        new_actions = {}
        if isinstance(actions, np.int64):
            new_actions["drone_1"] = self.air_actions[actions]
        else:
            new_actions = {}
            for key, value in actions.items():
                new_actions[key] = self.air_actions[value]

        states, rewards, truncated, dones, infos = super().step(new_actions)
        feature_set = self.state_wrapper(state=states, actions=new_actions, infos=infos)  # 处理 state

        new_rewards, new_dones, update_action = self.reward_wrapper(states=states, infos=infos, dones=dones,
                                                                    actions=new_actions)  # 处理 reward
        return_reward = 0
        if update_action is not None: # 遇到边界或snir较低的区域，回退一步
            self.re_step_count += 1  # 记录回退的次数
            return_reward = new_rewards
            states, rewards, truncated, dones, infos = super().step({"drone_1": update_action}) #回退一步
            feature_set = self.state_wrapper(state=states, actions=new_actions, infos=infos)  # 处理 state
            new_rewards, new_dones, update_action = self.reward_wrapper(states=states, infos=infos, dones=dones,
                                                                        actions=new_actions)  # 处理 reward
            if update_action is not None :  # 如果回退一步依然在sinr低的区域
                raise TypeError("Return to low SINR area, check the postion")


        if new_dones:
            self.actru_trajectory = self.ac_history_pos

        return feature_set, new_rewards+return_reward, new_dones, new_dones, infos

    def close(self) -> None:
        return super().close()
