import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def plot_train_graph(schedule, n_machines, n_jobs, save_path='train_graph.png'):
    plt.figure(figsize=(15, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, n_jobs))

    # 建立机器(Machine)到物理位置(Position)的映射
    # 车站映射: A=0, B=1, C=2, D=3, E=4
    machine_positions = {
        0: (0, 1),  # 下行: A->B
        1: (1, 2),  # 下行: B->C
        2: (2, 3),  # 下行: C->D
        3: (3, 4),  # 下行: D->E
        4: (4, 3),  # 上行: E->D
        5: (3, 2),  # 上行: D->C
        6: (2, 1),  # 上行: C->B
        7: (1, 0)  # 上行: B->A
    }

    for job_id in range(n_jobs):
        job_ops = []
        for (j, op), (m, start, end) in schedule.items():
            if j == job_id:
                job_ops.append((op, m, start, end))

        job_ops.sort(key=lambda x: x[0])
        times = []
        positions = []

        for op, m, start, end in job_ops:
            p_start, p_end = machine_positions[m]
            # 记录起点时间和位置
            times.append(start)
            positions.append(p_start)
            # 记录终点时间和位置
            times.append(end)
            positions.append(p_end)

        # 绘制折线
        actual_train_name = f'T{job_id + 1}'  # 车次名从 T1 开始

        if job_id in [8, 9]:  # T9 和 T10 是天窗列车
            for i in range(0, len(times), 2):
                t_start = times[i]
                t_end = times[i + 1]
                p_start = positions[i]
                p_end = positions[i + 1]

                label_str = f'Skylight ({actual_train_name})' if i == 0 else ""
                plt.plot([t_start, t_end], [p_start, p_end], '-', color='gray', linewidth=10, alpha=0.3,
                         label=label_str)
                plt.plot([t_start, t_end], [p_start, p_end], '--', color='black', linewidth=1)
        else:
            # 普通列车
            plt.plot(times, positions, 'o-', label=actual_train_name, color=colors[job_id], linewidth=2, markersize=4)
            if times:
                plt.text(times[0], positions[0], actual_train_name, verticalalignment='bottom', fontsize=9,
                         fontweight='bold')

    ticks = np.arange(0, 25 * 60, 60)
    tick_labels = [f'{int(t // 60)}h' for t in ticks]
    plt.xticks(ticks, tick_labels, rotation=45)
    plt.xlim(0, 24 * 60)

    plt.xlabel('Time (Hours)')
    plt.ylabel('Station')
    plt.title('Double-Track Train Working Diagram (Time-Distance Graph)')
    plt.grid(True, linestyle='--', alpha=0.7)

    station_names = ['A', 'B', 'C', 'D', 'E']
    plt.yticks(range(len(station_names)), station_names)

    # 将图例放在外面防止遮挡
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Train graph saved to {save_path}")
    plt.close()


def plot_gantt_chart(schedule, n_machines, n_jobs, save_path='gantt_chart.png'):
    """
    绘制资源占用甘特图
    Y轴：机器（车站/区间）
    X轴：时间
    """
    plt.figure(figsize=(15, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, n_jobs))

    # 显式标明各个机器代表的区间，让显示更直观
    machine_names = {
        0: 'M0 (A->B)', 1: 'M1 (B->C)', 2: 'M2 (C->D)', 3: 'M3 (D->E)',
        4: 'M4 (E->D)', 5: 'M5 (D->C)', 6: 'M6 (C->B)', 7: 'M7 (B->A)'
    }

    for (job_id, op_idx), (machine_id, start, end) in schedule.items():
        duration = end - start
        plt.barh(machine_id, duration, left=start, height=0.6,
                 color=colors[job_id], edgecolor='black', alpha=0.8)

        # 在条形图中间标注车次
        if duration > 2:  # 只有足够宽才标注
            # 【修复偏移】：真实显示的列车名应当是 job_id + 1
            plt.text(start + duration / 2, machine_id, f'T{job_id + 1}',
                     ha='center', va='center', color='white', fontsize=8)

    plt.xlabel('Time')
    plt.ylabel('Resource (Station / Section)')
    plt.title('Resource Occupation Gantt Chart')

    # 将Y轴标签替换为带方向的机器名字
    plt.yticks(range(n_machines), [machine_names[i] for i in range(n_machines)])
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Gantt chart saved to {save_path}")
    plt.close()


def save_schedule_to_csv(schedule, filename='results/train_schedule.csv'):
    """
    将调度结果保存为CSV（替代Excel以减少依赖）
    """
    machine_names = {
        0: 'M0 (A->B)', 1: 'M1 (B->C)', 2: 'M2 (C->D)', 3: 'M3 (D->E)',
        4: 'M4 (E->D)', 5: 'M5 (D->C)', 6: 'M6 (C->B)', 7: 'M7 (B->A)'
    }
    data = []
    for (job_id, op_idx), (machine_id, start, end) in schedule.items():
        data.append({
            'Train ID': f'T{job_id + 1}',  # 【修复偏移】
            'Operation ID': op_idx,
            'Station/Section (Machine)': machine_names[machine_id],  # 【加强直观性】
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
    df.to_csv(filename, index=False, encoding='utf_8_sig')
    print(f"Schedule saved to {filename}")