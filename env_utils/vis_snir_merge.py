'''
@Author: WANG Maonan
@Date: 2024-05-29 17:27:13
@Description: 对 SNIR 进行可视化
@LastEditTime: 2024-05-29 18:45:01
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def render_map(
        x_min, y_min, x_max, y_max, resolution, grid_z,
        trajectories_rl, trajectories_TSP, trajectories_PDPCC,figure_step, goal_points, speed, snir_threshold,
        img_path, figure_type, save_forbidden_area=False
):
    fig, ax = plt.subplots()

    start_points = []  # 乘客起点
    end_points = []  # 乘客终点
    for start, end in goal_points:
        start_points.append(start)
        end_points.append(end)

    new_grid_z, forbidden_area = generate_new_grid_z(0, 0, x_max, y_max, resolution, grid_z, speed, snir_threshold)

    if figure_type == 'step':
        plot_snir_binary(ax, 0, 0, x_max, y_max, snir_threshold, new_grid_z)  # 绘制 黑白SNIR

        plot_trajectories_step(ax, trajectories_rl, trajectories_TSP, trajectories_PDPCC, figure_step)  # 绘制 aircraft 的轨迹
        plot_goal_points(ax, start_points, color='#1e90ff', marker=">", point_type="S")  # 绘制乘客起点
        plot_goal_points(ax, end_points, color='#1e90ff', marker="o", point_type="D")  # 绘制乘客终点

    else:
        # 绘制对照组或实验组的结果
        plot_snir_map(ax, 0, 0, x_max, y_max, snir_threshold, new_grid_z)  # 绘制 SNIR

        plot_trajectories_sinr(ax, trajectories_rl, trajectories_TSP, trajectories_PDPCC, compare_flag=figure_type)  # 绘制 aircraft 的轨迹
        plot_goal_points(ax, start_points, color='#1e90ff', marker=">", point_type="S")  # 绘制乘客起点
        plot_goal_points(ax, end_points, color='#1e90ff', marker="o", point_type="D")  # 绘制乘客终点

    # 保存图像
    plt.savefig(img_path, dpi=300, bbox_inches='tight')

    # 显示图像
    plt.show()

    # 保存禁止区域的坐标
    if save_forbidden_area: # 保存列表到文件
        # 矩阵坐标 -> 地理系坐标（左下角为原点）
        new_forbidden_area = []
        for point in forbidden_area:
            new_forbidden_area.append((point[1], point[0]))
        np.savetxt(f'forbidden_area_{snir_threshold}.txt', new_forbidden_area)# 保存列表到文件


def generate_new_grid_z(x_min, y_min, x_max, y_max, resolution, grid_z, speed, snir_threshold):
    _x_max, _y_max = int((x_max - x_min) // resolution), int((y_max - y_min) // resolution)
    # _x_max, _y_max = grid_z.shape
    # 合并小格作为一个大格
    large_grid = int(speed / resolution)  # 一个大格里包含多少个小格
    new_grid_z = np.zeros((_x_max, _y_max))
    forbidden_area = []

    for x in range(0, _x_max, large_grid):
        for y in range(0, _y_max, large_grid):
            snir = []
            for k in range(x, x + large_grid):
                for l in range(y, y + large_grid):
                    if k < _x_max and l < _y_max:
                        snir.append(grid_z[k, l])
            mean_snir = np.nanmean(snir)
            min_snir = np.nanmin(grid_z)
            if mean_snir < snir_threshold:
                mean_snir = snir_threshold
                forbidden_area.append((x//10, y//10))
            if x + large_grid < _x_max and y + large_grid < _y_max:
                new_grid_z[x:x + large_grid, y:y + large_grid] = mean_snir
            elif x + large_grid < _x_max and y + large_grid >= _y_max:
                new_grid_z[x:x + large_grid, y:_y_max] = mean_snir
            elif x + large_grid >= _x_max and y + large_grid < _y_max:
                new_grid_z[x:_x_max, y:y + large_grid] = mean_snir
            else:
                new_grid_z[x:_x_max, y:_y_max] = mean_snir

    return new_grid_z, forbidden_area


def plot_snir_binary(ax, x_min, y_min, x_max, y_max, threshold, grid_z):
    """绘制 SNIR 的二值图
    """
    # 计算 x 和 y 坐标
    grid_z_masked = np.where(grid_z <= threshold, -15, grid_z)
    cmap = ListedColormap(['black', 'white'])
    bounds = [-15, threshold, 15]
    norm = BoundaryNorm(bounds, cmap.N)

    # 使用 imshow 显示数据，并设置自定义颜色映射和归一化
    cax = ax.imshow(grid_z_masked, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap=cmap,
                    norm=norm)

    # 添加颜色条
    cbar = plt.colorbar(cax, ax=ax, ticks=[-15, threshold, 15])
    cbar.ax.set_yticklabels(['-15', str(threshold), '15'])


def plot_snir_map(ax, x_min, y_min, x_max, y_max, threshold, grid_z):
    """绘制 SNIR 的底图
    """
    # 设置 NaN 的颜色为黑色
    grid_z_masked = np.where(grid_z <= threshold, np.nan, grid_z)
    cmap = plt.cm.viridis
    cmap.set_bad(color='black')

    # 绘制 grid_z 数值
    cax = ax.imshow(grid_z_masked, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap=cmap)  # 能和sinr原始底图对应
    # 修改颜色低透明度
    cax.set_alpha(0.6)

    # 添加颜色条
    plt.colorbar(cax, ax=ax)


def plot_trajectories_step(ax, trajectories_rl, trajectories_TSP, trajectories_PDPCC, figure_step):
    """绘制车辆轨迹信息
    """
    # first = 0
    path_rl = list(trajectories_rl.values())[0]
    path_TSP = list(trajectories_TSP.values())[0]
    path_PDPCC = list(trajectories_PDPCC.values())[0]
    # 提取 x 和 y 坐标
    x_coords_rl, y_coords_rl = zip(*path_rl)
    x_coords_TSP, y_coords_TSP = zip(*path_TSP)
    x_coords_PDPCC, y_coords_PDPCC = zip(*path_PDPCC)


    # 绘制轨迹的第一个点
    ax.scatter(x_coords_rl[0], y_coords_rl[0], s=140, c='#1e90ff', marker='*')

    # 点的附近加一个标签
    ax.annotate("UAM start", xy=(x_coords_rl[0], y_coords_rl[0]), xytext=(x_coords_rl[0] + 50, y_coords_rl[0] + 50), fontsize=8)

    # 绘制轨迹
    color = ["#253494", "#e34a33", "#006d2c"] # #253494深蓝色

    # ax.plot(x_coords_rl, y_coords_rl, c=color[1], label="MSHA-RL")
    ax.plot(x_coords_PDPCC, y_coords_PDPCC, c=color[0], label="PDPCC")
    # ax.plot(x_coords_TSP, y_coords_TSP, c=color[2], label="CPTSP", linewidth=3, linestyle='--')

    # ax.scatter(x_coords_rl[-1], y_coords_rl[-1], s=140, c=color[1], marker='*')
    ax.scatter(x_coords_PDPCC[-1], y_coords_PDPCC[-1], s=180, c=color[0], marker='*')
    # ax.scatter(x_coords_TSP[-1], y_coords_TSP[-1], s=120, c=color[2], marker='*')


    # 添加图例
    ax.legend()

def plot_trajectories_sinr(ax, trajectories_rl, trajectories_TSP, trajectories_PDPCC, compare_flag):
    """绘制车辆轨迹信息
    """
    path_rl = list(trajectories_rl.values())[0]
    path_TSP = list(trajectories_TSP.values())[0]
    path_PDPCC = list(trajectories_PDPCC.values())[0]
    # 提取 x 和 y 坐标
    x_coords_rl, y_coords_rl = zip(*path_rl)
    x_coords_TSP, y_coords_TSP = zip(*path_TSP)
    x_coords_PDPCC, y_coords_PDPCC = zip(*path_PDPCC)
    x_coords_stright, y_coords_stright = ([2600, 2220, 1850, 1450, 1010, 400, 650, 1270, 1700],
                                          [1500, 1400, 1100, 2000, 2220, 1200, 1000, 450, 620])

    # 绘制轨迹的第一个点
    ax.scatter(x_coords_rl[0], y_coords_rl[0], s=140, c='#1e90ff', marker='*')

    # 点的附近加一个标签
    ax.annotate("UAM start", xy=(x_coords_rl[0], y_coords_rl[0]), xytext=(x_coords_rl[0] + 50, y_coords_rl[0] + 50),
                fontsize=8)

    color = ["#253494", "#e34a33", "#006d2c"]  # #253494深蓝色
    # color = ["#30ad8c", "#d28ec6", "#2f8abe"]  # #253494深蓝色
    # 绘制轨迹
    if compare_flag=="com": # exp: RL+straight, com: TSP+PDPCC, com_tsp:TSP, com_pdp:PDPCC
        ax.plot(x_coords_PDPCC, y_coords_PDPCC, c=color[0], label="PDPCC")
        ax.plot(x_coords_TSP, y_coords_TSP, c=color[2], label="CPTSP", linewidth=3, linestyle='--')
        # 绘制轨迹的最后一个点
        ax.scatter(x_coords_PDPCC[-1], y_coords_PDPCC[-1], s=180, c=color[0], marker='*')
        ax.scatter(x_coords_TSP[-1], y_coords_TSP[-1], s=120, c=color[2], marker='*')

    elif compare_flag=="exp":
        ax.plot(x_coords_rl, y_coords_rl, c=color[1], label="MSHA-RL")
        ax.plot(x_coords_stright, y_coords_stright, c='black', label="Stright Line", linewidth=2, linestyle='--')

        ax.scatter(x_coords_rl[-1], y_coords_rl[-1], s=140, c=color[1], marker='*')

    elif compare_flag=="com_tsp":
        ax.plot(x_coords_TSP, y_coords_TSP, c=color[2], label="CPTSP", linewidth=3, linestyle='--')
        # 绘制轨迹的最后一个点
        ax.scatter(x_coords_TSP[-1], y_coords_TSP[-1], s=120, c=color[2], marker='*')
    elif compare_flag=="com_pdp":
        ax.plot(x_coords_PDPCC, y_coords_PDPCC, c=color[0], label="PDPCC")
        # 绘制轨迹的最后一个点
        ax.scatter(x_coords_PDPCC[-1], y_coords_PDPCC[-1], s=180, c=color[0], marker='*')


    # 添加图例
    ax.legend()

def plot_goal_points(ax, goal_points, color, marker, point_type):
    """绘制目标点
    """
    if point_type == "S":
        label = "Passenger Start"
    else:
        label = "Passenger Destination"

    for idx, point in enumerate(goal_points):
        ax.scatter(*point, s=50, c=color, marker=marker, label=f"{point_type}{idx} at {point}")
        # 每一个点旁边加一个标签
        # if idx == 0 and point_type == "S":
        #     ax.annotate(f"{point_type}{idx+1}", xy=point,
        #                 xytext=(point[0]-20, point[1] +int(100)))
        # elif idx == 0 and point_type == "D":
        #     ax.annotate(f"{point_type}{idx+1}", xy=point,
        #                 xytext=(point[0] + int(50), point[1] ))
        #
        # else:
        #     ax.annotate(f"{point_type}{idx+1}", xy=point, xytext=(point[0] + int(50), point[1] + int(50)))
        # ax.annotate(f"{point_type}{idx}", xy=point, xytext=(point[0] + int(50), point[1] + int(50)))

    # 添加图例
    # ax.legend()

def get_curve_function(p1,p2):
    # return lambda x: (p2[1]-p1[1])/(p2[0]-p1[0])*(x-p1[0])+p1[1]
    return lambda y: (p2[0]-p1[0])/(p2[1]-p1[1])*(y-p1[1])+p1[0]

# if __name__ == '__main__':
#     f = get_curve_function((1,0),(0,2))
#     print(f(1))
