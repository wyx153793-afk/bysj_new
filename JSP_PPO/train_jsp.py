"""
JSP 训练脚本（单动作版本）
适用于高铁运行图编制问题
"""
import torch
import numpy as np
import os
import glob
import re
from datetime import datetime
from JSP_Env import JSP_Env
from PPO_JSP import PPO_JSP
import matplotlib.pyplot as plt
from vis_utils import plot_train_graph, plot_gantt_chart, save_schedule_to_csv

def get_manual_data():
    """手动输入数据示例：双线高铁场景"""
    distance_ab = 30
    distance_bc = 32
    distance_cd = 38
    distance_de = 21

    normal_train_speed = 2.0  # 普通列车 120 km/h = 2 km/min
    skylight_total_time = 240 # 4 小时 = 240 分钟

    t_ab = int(distance_ab / normal_train_speed)
    t_bc = int(distance_bc / normal_train_speed)
    t_cd = int(distance_cd / normal_train_speed)
    t_de = int(distance_de / normal_train_speed)

    total_dist = distance_ab + distance_bc + distance_cd + distance_de
    t_ab_w = int((distance_ab / total_dist) * skylight_total_time)
    t_bc_w = int((distance_bc / total_dist) * skylight_total_time)
    t_cd_w = int((distance_cd / total_dist) * skylight_total_time)
    t_de_w = skylight_total_time - t_ab_w - t_bc_w - t_cd_w

    n_jobs = 10
    n_machines = 8
    max_ops = 4

    processing_time = np.zeros((n_jobs, max_ops), dtype=int)
    op_machine_assign = -np.ones((n_jobs, max_ops), dtype=int)

    # === 下行列车单数定制径路 (Downbound) ===
    # T1(Job 0): A->B
    processing_time[0, 0] = t_ab; op_machine_assign[0, 0] = 0
    # T3(Job 2): A->C
    processing_time[2, 0] = t_ab; op_machine_assign[2, 0] = 0
    processing_time[2, 1] = t_bc; op_machine_assign[2, 1] = 1
    # T5(Job 4): B->E (从B站发车)
    processing_time[4, 0] = t_bc; op_machine_assign[4, 0] = 1
    processing_time[4, 1] = t_cd; op_machine_assign[4, 1] = 2
    processing_time[4, 2] = t_de; op_machine_assign[4, 2] = 3
    # T7(Job 6): A->E
    processing_time[6, 0] = t_ab; op_machine_assign[6, 0] = 0
    processing_time[6, 1] = t_bc; op_machine_assign[6, 1] = 1
    processing_time[6, 2] = t_cd; op_machine_assign[6, 2] = 2
    processing_time[6, 3] = t_de; op_machine_assign[6, 3] = 3

    # T9(Job 8): 下行天窗列车 A->E
    processing_time[8, 0] = t_ab_w; op_machine_assign[8, 0] = 0
    processing_time[8, 1] = t_bc_w; op_machine_assign[8, 1] = 1
    processing_time[8, 2] = t_cd_w; op_machine_assign[8, 2] = 2
    processing_time[8, 3] = t_de_w; op_machine_assign[8, 3] = 3

    # === 上行列车双数定制径路 (Upbound) ===
    # T2(Job 1): E->D
    processing_time[1, 0] = t_de; op_machine_assign[1, 0] = 4
    # T4(Job 3): E->C
    processing_time[3, 0] = t_de; op_machine_assign[3, 0] = 4
    processing_time[3, 1] = t_cd; op_machine_assign[3, 1] = 5
    # T6(Job 5): D->A
    processing_time[5, 0] = t_cd; op_machine_assign[5, 0] = 5
    processing_time[5, 1] = t_bc; op_machine_assign[5, 1] = 6
    processing_time[5, 2] = t_ab; op_machine_assign[5, 2] = 7
    # T8(Job 7): E->A
    processing_time[7, 0] = t_de; op_machine_assign[7, 0] = 4
    processing_time[7, 1] = t_cd; op_machine_assign[7, 1] = 5
    processing_time[7, 2] = t_bc; op_machine_assign[7, 2] = 6
    processing_time[7, 3] = t_ab; op_machine_assign[7, 3] = 7

    # T10(Job 9): 上行天窗列车 E->A
    processing_time[9, 0] = t_de_w; op_machine_assign[9, 0] = 4
    processing_time[9, 1] = t_cd_w; op_machine_assign[9, 1] = 5
    processing_time[9, 2] = t_bc_w; op_machine_assign[9, 2] = 6
    processing_time[9, 3] = t_ab_w; op_machine_assign[9, 3] = 7

    return processing_time, op_machine_assign, n_jobs, n_machines, max_ops

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

    # 确定当前训练的最优模型保存路径
    current_best_dir = dir_fixed if USE_FIXED_EPISODES else dir_infinite

    print(f"Training Mode: {'Fixed Episodes' if USE_FIXED_EPISODES else 'Train Until Optimal'}")
    print(f"Current best models will be saved to: {current_best_dir}")

    proc_times, machine_assign, n_jobs, n_machines, max_ops = get_manual_data()
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

    # 用于追踪本轮训练的最优记录和对应的文件路径
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

        env = JSP_Env(n_jobs, n_machines, proc_times, machine_assign, device=device)
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

            # 【核心修改点】：删除这轮训练中刚刚产生的“上一代”最优模型，确保本轮最后只留一个
            if session_best_model_path and os.path.exists(session_best_model_path):
                try:
                    os.remove(session_best_model_path)
                except OSError as e:
                    print(f"Warning: Failed to delete previous best model file: {e}")

            # 保存新模型，并更新当前最新路径
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

        # 早停判断 (仅在无限制模式下生效)
        if not USE_FIXED_EPISODES and episodes_without_improvement >= patience:
            print(f"🔴 No improvement for {patience} episodes. Early stopping at optimal conditions.")
            break

        episode += 1

    # 训练结束，保存最后一步的策略
    current_model_path = os.path.join(results_dir, 'policy_job_final.pth')
    ppo.save(current_model_path)
    np.save(os.path.join(results_dir, 'training_history.npy'), makespan_history)

    # 绘制结果
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