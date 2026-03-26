"""
JSP 训练脚本（单动作版本）
"""
import torch
import numpy as np
import os
import glob
import re
import pandas as pd
from datetime import datetime
from JSP_Env import JSP_Env
from PPO_JSP import PPO_JSP
import matplotlib.pyplot as plt
from vis_utils import plot_train_graph, plot_gantt_chart, save_schedule_to_csv

def get_excel_data(excel_path='data/data2.xlsx'):
    """从 Excel 读取高铁调度基础数据"""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"找不到数据文件：{excel_path}。请确保存在 data 文件夹且文件名为 data*.xlsx")

    # 1. 读取全局参数 (Sheet1)
    df_global = pd.read_excel(excel_path, sheet_name='Sheet1').set_index('param_name')
    normal_train_speed = float(df_global.loc['normal_train_speed', 'param_value'])
    skylight_total_time = float(df_global.loc['skylight_total_time', 'param_value'])

    # 2. 读取区间信息 (Sheet2)，建立 机器ID -> 行驶时间 的映射
    df_sections = pd.read_excel(excel_path, sheet_name='Sheet2')
    total_dist = df_sections['distance'].sum()

    machine_time_map = {}
    machine_skylight_time_map = {}

    for _, row in df_sections.iterrows():
        dist = row['distance']
        # 普通列车时间
        normal_time = int(dist / normal_train_speed)
        # 天窗时间 (按距离比例分配)
        skylight_time = int((dist / total_dist) * skylight_total_time)

        # 下行映射
        machine_time_map[row['down_machine']] = normal_time
        machine_skylight_time_map[row['down_machine']] = skylight_time
        # 上行映射
        machine_time_map[row['up_machine']] = normal_time
        machine_skylight_time_map[row['up_machine']] = skylight_time

    # 3. 读取列车任务信息 (Sheet3)
    # 兼容可能存在的不同 Sheet 命名习惯
    try:
        df_trains = pd.read_excel(excel_path, sheet_name='Sheet3')
    except ValueError:
        try:
            df_trains = pd.read_excel(excel_path, sheet_name='sheet3')
        except ValueError:
            df_trains = pd.read_excel(excel_path, sheet_name='Trains')

    n_jobs = len(df_trains)

    # 序列解析函数，增强容错能力
    def parse_sequence(seq_val):
        seq_str = str(seq_val)
        # 替换中文逗号为英文逗号
        seq_str = seq_str.replace('，', ',')
        # 剔除所有空格
        seq_str = seq_str.replace(' ', '')

        # 拦截被 Excel 错误转成日期格式的数据 (如 2026-04-05)
        if '-' in seq_str or ':' in seq_str:
            raise ValueError(f"数据读取错误：识别到类似日期的格式 '{seq_str}'。请在 Excel 中将该列设置为'文本'格式，或输入时以单引号开头 (如 '4,5)。")

        return [int(m) for m in seq_str.split(',')]

    all_seqs = df_trains['machine_seq'].apply(parse_sequence)
    max_ops = all_seqs.apply(len).max()

    # 自动推算机器总数 (以出现的最大的机器ID+1为准，保证索引不越界)
    all_machines = set()
    for seq in all_seqs:
        all_machines.update(seq)
    n_machines = max(all_machines) + 1

    # 4. 初始化输出矩阵与天窗列表
    processing_time = np.zeros((n_jobs, max_ops), dtype=int)
    op_machine_assign = -np.ones((n_jobs, max_ops), dtype=int)
    maintenance_job_ids = []  # 【新增】动态收集天窗任务的 ID

    # 5. 填充矩阵数据并增加安全校验
    for idx, row in df_trains.iterrows():
        job_id = int(row['job_id'])
        is_skylight = (row['type'] == 'Skylight')

        # 【新增】如果是天窗任务，将 job_id 加入列表
        if is_skylight:
            maintenance_job_ids.append(job_id)

        seq = all_seqs.iloc[idx]

        for op_idx, m_id in enumerate(seq):
            if m_id not in machine_time_map:
                raise ValueError(f"配置错误：车次 {row['train_name']} 的机器编号 {m_id} 在区间配置(Sheet2)中不存在！")

            op_machine_assign[job_id, op_idx] = m_id
            if is_skylight:
                processing_time[job_id, op_idx] = machine_skylight_time_map[m_id]
            else:
                processing_time[job_id, op_idx] = machine_time_map[m_id]

    # 【修改】返回参数中加上 maintenance_job_ids
    return processing_time, op_machine_assign, n_jobs, n_machines, max_ops, maintenance_job_ids

def find_best_historical_model(directories):
    """
    扫描指定的文件夹列表，解析文件名，返回 makespan 最小的模型路径
    """
    best_path = None
    global_best_makespan = float('inf')
    # 正则匹配文件名中的 makespan 数值
    pattern = re.compile(r'jsp_best_mk_([0-9.]+)_')

    for directory in directories:
        if not os.path.exists(directory):
            continue
        for filepath in glob.glob(os.path.join(directory, '*.pth')):
            filename = os.path.basename(filepath)
            match = pattern.search(filename)
            if match:
                mk_val = float(match.group(1))
                # 寻找 makespan 最小的（时间越短越优）
                if mk_val < global_best_makespan:
                    global_best_makespan = mk_val
                    best_path = filepath

    return best_path, global_best_makespan

def train():
    # ==========================================
    # 模式切换区
    # ==========================================
    USE_FIXED_EPISODES = False  # True: 固定轮数; False: 无限轮数到最优即停
    n_episodes = 500           # 固定模式下的总轮数
    patience = 50              # 无限模式下：连续多少轮没破记录则早停

    # 自动加载历史最优模型开关
    LOAD_PREVIOUS_BEST = True
    # ==========================================

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/train_{timestamp}'

    # 为两种训练模式分配不同的最优模型文件夹
    dir_fixed = 'results/best_models_fixed'
    dir_infinite = 'results/best_models_infinite'

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(dir_fixed, exist_ok=True)
    os.makedirs(dir_infinite, exist_ok=True)

    current_best_dir = dir_fixed if USE_FIXED_EPISODES else dir_infinite

    print(f"Training Mode: {'Fixed Episodes' if USE_FIXED_EPISODES else 'Train Until Optimal'}")
    print(f"Current best models will be saved to: {current_best_dir}")

    # 调用修改后的 get_excel_data 函数
    try:
        proc_times, machine_assign, n_jobs, n_machines, max_ops, maintenance_job_ids = get_excel_data(
            'data/data2.xlsx')
        print(f"成功加载 Excel 数据！总任务数: {n_jobs}, 机器数: {n_machines}")
        print(f"识别到天窗列车 ID: {maintenance_job_ids}")  # 打印出来确认一下
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ppo = PPO_JSP(n_jobs, n_machines, device=device)

    # ==========================================
    # 调用二者中更优的那个模型
    # ==========================================
    if LOAD_PREVIOUS_BEST:
        best_hist_path, best_hist_mk = find_best_historical_model([dir_fixed, dir_infinite])
        if best_hist_path:
            print(f"🚀 Found historical best model across all modes!")
            print(f"   Path: {best_hist_path}")
            print(f"   Best Makespan: {best_hist_mk}")
            try:
                ppo.load(best_hist_path)
                print("   Successfully loaded! Continuing training from this elite state...\n")
            except Exception as e:
                print(f"   Failed to load existing model: {e}\n")
        else:
            print("No historical models found. Starting training from scratch.\n")

    session_best_makespan = float('inf')
    session_best_model_path = None

    makespan_history = []
    last_env = None
    episodes_without_improvement = 0
    episode = 0

    while True:
        if USE_FIXED_EPISODES and episode >= n_episodes:
            print(f"Reached fixed maximum episodes ({n_episodes}). Stopping.")
            break

        # 【修改】将 maintenance_job_ids 传给环境
        env = JSP_Env(n_jobs, n_machines, proc_times, machine_assign,
                      maintenance_job_ids=maintenance_job_ids, device=device)
        state = env.reset()
        done = False

        while not done:
            action, log_prob = ppo.select_action(state)
            next_state, reward, done, info = env.step(action)
            ppo.store_transition(state, action, reward, next_state, done, log_prob)
            state = next_state

        last_env = env
        makespan = env.get_makespan()
        makespan_history.append(makespan)

        # 发现本次训练的新最优解
        if makespan < session_best_makespan:
            session_best_makespan = makespan
            episodes_without_improvement = 0

            new_save_path = os.path.join(current_best_dir, f'jsp_best_mk_{makespan}_{timestamp}.pth')

            if session_best_model_path and os.path.exists(session_best_model_path):
                try:
                    os.remove(session_best_model_path)
                except OSError as e:
                    print(f"Warning: Failed to delete previous best model file: {e}")

            ppo.save(new_save_path)
            session_best_model_path = new_save_path
            print(f"🟢 Episode {episode}: New session best makespan {makespan} found! Saved to {new_save_path}")
        else:
            episodes_without_improvement += 1

        if episode % 10 == 0 and episode > 0:
            ppo.update()

        if episode % 50 == 0:
            avg_makespan = np.mean(makespan_history[-50:]) if len(makespan_history) >= 50 else np.mean(makespan_history)
            print(f"Episode {episode}, Current: {makespan}, Avg(50): {avg_makespan:.2f}, No Improve: {episodes_without_improvement}")

        if not USE_FIXED_EPISODES and episodes_without_improvement >= patience:
            print(f"🔴 No improvement for {patience} episodes. Early stopping at optimal conditions.")
            break

        episode += 1

    current_model_path = os.path.join(results_dir, 'policy_job_final.pth')
    ppo.save(current_model_path)
    np.save(os.path.join(results_dir, 'training_history.npy'), makespan_history)

    if last_env:
        plot_train_graph(last_env.schedule, n_machines, n_jobs, os.path.join(results_dir, 'train_graph.png'))
        plot_gantt_chart(last_env.schedule, n_machines, n_jobs, os.path.join(results_dir, 'gantt_chart.png'))
        save_schedule_to_csv(last_env.schedule, os.path.join(results_dir, 'train_schedule.csv'))

    plt.figure()
    plt.plot(makespan_history)
    plt.xlabel('Episode')
    plt.ylabel('Makespan (Excluding Skylight)')
    plt.title(f'JSP Training Curve ({timestamp})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(results_dir, 'training_curve.png'))
    plt.close()

if __name__ == '__main__':
    train()