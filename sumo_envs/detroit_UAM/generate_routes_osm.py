'''
@Author: WANG Maonan
@Date: 2023-09-14 12:43:28
@Description: 给 OSM 环境生成 route 的例子
@LastEditTime: 2023-09-14 15:47:49
'''
import numpy as np
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from tshub.sumo_tools.generate_routes import generate_route

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'))

# 开启仿真 --> 指定 net 文件
# sumo_net = current_file_path("./ND_env/rectangle.net.xml")
# sumo_net = current_file_path("./US_UAM/US_UAM.net.xml")
sumo_net = current_file_path("./detroit_UAM/detroit_UAM.net.xml")

# 指定要生成的路口 id 和探测器保存的位置
generate_route(
    sumo_net=sumo_net,
    interval=[6, 6, 6], # 每个slot 2min
    edge_flow_per_minute={ # edge ID
        '8717259#0':[1,0,0],
        # 'E0': [0],# 每分钟20辆车
        # 'E1': [0, 4, 0],
        # 'E7': [2, 4, 0],
        # 'E15':[0, 2, 0],
        # 'E22': [0, 30, 0],
        # '1125695391#0.789': [np.random.randint(10, 15) for _ in range(4)],
        # '1125691753#1': [np.random.randint(10, 15) for _ in range(4)],
        # '1125790983#0': [np.random.randint(10, 15) for _ in range(4)],
        # '1125793701#3': [np.random.randint(10, 15) for _ in range(4)],
        # '219056200#2': [np.random.randint(10, 15) for _ in range(4)],
    }, # 每分钟每个 edge 有多少车
    edge_turndef={
        '8717259#0__449611417#0':[1,0,1],
    # 'E1': [10, 10, 0],
        # 'E24'
        # 'E0__E6': [0.5, 0.5, 0.5],
        # 'E0__E3': [0.5, 0.5, 0.5],
        # 'E1__E2': [0.5, 0.5, 0.5],
        # 'E1__E8': [0.5, 0.5, 0.5],
        # 'E2__E4': [0.5, 0.5, 0.5],
        # 'E2__E10': [0.5, 0.5, 0.5],
        # 'E3__E4': [0.5, 0.5, 0.5],
        # 'E3__E10': [0.5, 0.5, 0.5],
        # 'E4__E5': [0.5, 0.5, 0.5],
        # 'E4__E12': [0.5, 0.5, 0.5],
        # 'E7__E8': [0.5, 0.5, 0.5],
        # 'E7__E2': [0.5, 0.5, 0.5],
        # 'E8__E16': [0.5, 0.5, 0.5],
        # 'E8__E9': [0.5, 0.5, 0.5],
        # 'E11__E13': [0.5, 0.5, 0.5],
        # 'E11__E17': [0.5, 0.5, 0.5],
        # 'E12__E13': [0.5, 0.5, 0.5],
        # 'E12__E17': [0.5, 0.5, 0.5],
        # 'E15__E9': [0.5, 0.5, 0.5],
        # 'E15__E16': [0.5, 0.5, 0.5],

        # 'E8__E9': [0.5, 0.5],
        # 'E22__E0__E6__E1__E24__E22': [1,1,1],
        # '1125684496#1__1125597092#1': [0.7, 0.7, 0.8, 0.7],
        # '1125695392#0__1125695392#7.174': [0.7, 0.7, 0.8, 0.7],
        # '1125790983#2__1125684496#0': [0.7, 0.7, 0.8, 0.7]
    },
    # 设置汽车属性（速度）
    veh_type={
        # 'ego': {'color':'26, 188, 156', 'probability':0.3},
        'background': {'color':'155, 89, 182', 'probability':1, 'speed':3},
    },
    output_trip=current_file_path('./detroit_UAM/testflow.trip.xml'),
    output_turndef=current_file_path('./detroit_UAM/testflow.turndefs.xml'),
    output_route=current_file_path('./detroit_UAM/detroit_UAM.rou.xml'),
    interpolate_flow=False,
    interpolate_turndef=False,
)

