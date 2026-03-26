"""
Visualization Utilities for JSP
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl

# 尝试解决图表中文字体显示问题 (防止乱码)
try:
    mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    mpl.rcParams['axes.unicode_minus'] = False
except:
    pass

def get_station_name(i):
    """动态获取车站名，如 0->A, 1->B ... 26->S26"""
    if i < 26:
        return chr(65 + i)
    return f'S{i}'

def plot_train_graph(schedule, n_machines, n_jobs, save_path):
    """
    绘制高铁运行图 (Time-Distance Graph)
    """
    plt.figure(figsize=(15, 8))

    n_sections = n_machines // 2
    n_stations = n_sections + 1

    # 动态生成颜色
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, n_jobs)))

    # 动态建立机器(Machine)到物理位置(Position)的映射
    machine_positions = {}
    for m in range(n_sections):
        machine_positions[m] = (m, m + 1)            # 下行
        opp_m = n_machines - 1 - m
        machine_positions[opp_m] = (m + 1, m)        # 上行

    for job_id in range(n_jobs):
        job_ops = []
        for (j, op), (m, start, end) in schedule.items():
            if j == job_id:
                job_ops.append((op, m, start, end))

        if not job_ops:
            continue

        job_ops.sort(key=lambda x: x[0])
        times = []
        positions = []

        for op, m, start, end in job_ops:
            if m not in machine_positions:
                continue
            p_start, p_end = machine_positions[m]
            # 记录起点时间和位置
            times.append(start)
            positions.append(p_start)
            # 记录终点时间和位置
            times.append(end)
            positions.append(p_end)

        # 绘制折线
        actual_train_name = f'T{job_id + 1}'  # 车次名从 T1 开始

        # 天窗列车样式 (保留目标代码中对特定job_id的特判，如8,9)
        if job_id in [8, 9]:
            for i in range(0, len(times), 2):
                t_start = times[i]
                t_end = times[i + 1]
                p_start = positions[i]
                p_end = positions[i + 1]

                label_str = f'Skylight ({actual_train_name})' if i == 0 else ""
                plt.plot([t_start, t_end], [p_start, p_end], '-', color='gray', linewidth=10, alpha=0.3, label=label_str)
                plt.plot([t_start, t_end], [p_start, p_end], '--', color='black', linewidth=1)
        else:
            # 普通列车
            plt.plot(times, positions, 'o-', label=actual_train_name, color=colors[job_id % len(colors)], linewidth=2, markersize=4)
            if times:
                plt.text(times[0], positions[0], actual_train_name, verticalalignment='bottom', fontsize=9, fontweight='bold')

    # X轴刻度：按小时划分，展示 0h, 1h...
    ticks = np.arange(0, 25 * 60, 60)
    tick_labels = [f'{int(t // 60)}h' for t in ticks]
    plt.xticks(ticks, tick_labels, rotation=45)
    plt.xlim(0, 24 * 60)

    plt.xlabel('Time (Hours)', fontsize=14)
    plt.ylabel('Station', fontsize=14)
    plt.title('High-Speed Railway Train Graph (运行图)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Y轴刻度：动态生成车站名称 A, B, C...
    station_names = [get_station_name(i) for i in range(n_stations)]
    plt.yticks(range(n_stations), station_names)

    # 将图例放在外面防止遮挡
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_gantt_chart(schedule, n_machines, n_jobs, save_path):
    """
    绘制资源占用甘特图
    Y轴：机器（车站/区间）
    X轴：时间
    """
    plt.figure(figsize=(15, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, n_jobs)))

    n_sections = n_machines // 2
    n_stations = n_sections + 1
    station_names = [get_station_name(i) for i in range(n_stations)]

    # 动态显式标明各个机器代表的区间，增强直观性
    machine_names = {}
    for m in range(n_sections):
        machine_names[m] = f'M{m} ({station_names[m]}->{station_names[m+1]})'
        opp_m = n_machines - 1 - m
        machine_names[opp_m] = f'M{opp_m} ({station_names[m+1]}->{station_names[m]})'

    for (job_id, op_idx), (machine_id, start, end) in schedule.items():
        duration = end - start
        plt.barh(machine_id, duration, left=start, height=0.6,
                 color=colors[job_id % len(colors)], edgecolor='black', alpha=0.8)

        # 在条形图中间标注车次
        if duration > 2:
            plt.text(start + duration / 2, machine_id, f'T{job_id + 1}',
                     ha='center', va='center', color='white', fontsize=8)

    plt.xlabel('Time (minutes)', fontsize=14)
    plt.ylabel('Resource (Station / Section)', fontsize=14)
    plt.title('Resource Occupation Gantt Chart', fontsize=16)

    # 将Y轴标签替换为带方向的机器名字
    plt.yticks(range(n_machines), [machine_names.get(i, f'M{i}') for i in range(n_machines)])
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_schedule_to_csv(schedule, save_path):
    """
    将调度结果保存为CSV
    """
    if not schedule:
        pd.DataFrame().to_csv(save_path, index=False)
        return

    # 推算并动态生成机器名称 (以适配不同数量的区间)
    n_machines = max([m_id for (_, _), (m_id, _, _) in schedule.items()]) + 1
    n_sections = n_machines // 2
    n_stations = n_sections + 1
    station_names = [get_station_name(i) for i in range(n_stations)]

    machine_names = {}
    for m in range(n_sections):
        machine_names[m] = f'M{m} ({station_names[m]}->{station_names[m+1]})'
        opp_m = n_machines - 1 - m
        machine_names[opp_m] = f'M{opp_m} ({station_names[m+1]}->{station_names[m]})'

    data = []
    for (job_id, op_idx), (machine_id, start, end) in schedule.items():
        data.append({
            'Train ID': f'T{job_id + 1}',
            'Operation ID': op_idx,
            'Station/Section (Machine)': machine_names.get(machine_id, f'M{machine_id}'),
            'Start Time': start,
            'End Time': end,
            'Duration': end - start
        })

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 按车次和开始时间排序 (需要转换为整数排序保证 T10 在 T2 后面)
    df['TrainNum'] = df['Train ID'].apply(lambda x: int(x[1:]))
    df = df.sort_values(by=['TrainNum', 'Start Time']).drop('TrainNum', axis=1)

    # 保存为CSV
    df.to_csv(save_path, index=False, encoding='utf_8_sig')