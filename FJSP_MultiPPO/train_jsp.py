"""
JSP 训练脚本（单动作版本）
适用于高铁运行图编制问题
"""
import torch
import numpy as np
import os
from datetime import datetime
from JSP_Env import JSP_Env
from PPO_JSP import PPO_JSP
import matplotlib.pyplot as plt
from vis_utils import plot_train_graph, plot_gantt_chart, save_schedule_to_csv

def get_manual_data():
    """
    手动输入数据示例
    用户可以在这里修改列车开行方案和技术参数
    
    Returns:
        processing_time: [n_jobs, max_ops] 加工时间
        op_machine_assign: [n_jobs, max_ops] 机器分配
    """
    # 场景设置：
    # 5 列车（T0-T3 为普通，T4 为天窗）
    # 5 车站：A(0), B(1), C(2), D(3), E(4)
        
    # ========== 用户输入参数 ==========
    # 区间间距 (km)
    distance_ab = 30  # A 到 B 的距离
    distance_bc = 32  # B 到 C 的距离
    distance_cd = 38  # C 到 D 的距离
    distance_de = 21  # D 到 E 的距离
        
    # 列车运行速度 (km/min)
    normal_train_speed = 2.0  # 普通列车速度 120 km/h = 2 km/min
        
    # 天窗列车总作业时间约束 (分钟)
    skylight_total_time = 240  # 4 小时 = 240 分钟
    # ===================================
        
    # 自动计算各区间的运行时间
    # 公式：时间 = 距离 / 速度
    t_ab = int(distance_ab / normal_train_speed)
    t_bc = int(distance_bc / normal_train_speed)
    t_cd = int(distance_cd / normal_train_speed)
    t_de = int(distance_de / normal_train_speed)
        
    # 天窗列车按距离比例分配各区间作业时间
    total_distance_ad = distance_ab + distance_bc + distance_cd
    t_ab_work = int((distance_ab / total_distance_ad) * skylight_total_time)
    t_bc_work = int((distance_bc / total_distance_ad) * skylight_total_time)
    # 调整最后一个区间的时间，确保总和正好为指定时间
    t_cd_work = skylight_total_time - t_ab_work - t_bc_work
    
    # 场景基本参数
    n_jobs = 5  # 5 列车
    n_machines = 4  # 4 个区间
    max_ops = 4  # 最多 4 个工序
    
    # 机器 ID 定义：0(A-B), 1(B-C), 2(C-D), 3(D-E)
    
    processing_time = np.zeros((n_jobs, max_ops), dtype=int)
    op_machine_assign = -np.ones((n_jobs, max_ops), dtype=int)
    
    # T0: A->B
    processing_time[0, 0] = t_ab
    op_machine_assign[0, 0] = 0
    
    # T1: A->C
    processing_time[1, 0] = t_ab; op_machine_assign[1, 0] = 0
    processing_time[1, 1] = t_bc; op_machine_assign[1, 1] = 1
    
    # T2: B->E
    processing_time[2, 0] = t_bc; op_machine_assign[2, 0] = 1
    processing_time[2, 1] = t_cd; op_machine_assign[2, 1] = 2
    processing_time[2, 2] = t_de; op_machine_assign[2, 2] = 3
    
    # T3: A->E
    processing_time[3, 0] = t_ab; op_machine_assign[3, 0] = 0
    processing_time[3, 1] = t_bc; op_machine_assign[3, 1] = 1
    processing_time[3, 2] = t_cd; op_machine_assign[3, 2] = 2
    processing_time[3, 3] = t_de; op_machine_assign[3, 3] = 3
    
    # T4: A->D (天窗)
    processing_time[4, 0] = t_ab_work; op_machine_assign[4, 0] = 0
    processing_time[4, 1] = t_bc_work; op_machine_assign[4, 1] = 1
    processing_time[4, 2] = t_cd_work; op_machine_assign[4, 2] = 2
    
    return processing_time, op_machine_assign, n_jobs, n_machines, max_ops


def train():
    # 参数设置
    # 已设置为强制使用手动输入的数据（5 列车，5 车站场景）
    USE_MANUAL_DATA = True
    
    # 生成当前训练的时间戳作为唯一标识
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建以时间戳命名的结果文件夹
    results_dir = f'results/train_{timestamp}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    print(f"Training results will be saved to: {results_dir}")
    
    # 保存配置文件到结果目录（可选）
    config_file = os.path.join(results_dir, 'config.txt')
    with open(config_file, 'w') as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"N Jobs: 5\n")
        f.write(f"N Machines: 4\n")
        f.write(f"N Episodes: 500\n")
        f.write(f"Start Time: {datetime.now()}\n")

    if USE_MANUAL_DATA:
        proc_times, machine_assign, n_jobs, n_machines, max_ops = get_manual_data()
        n_episodes = 500 # 保持一定训练量以优化
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 初始化 PPO
    ppo = PPO_JSP(n_jobs, n_machines, device=device)
    
    # 检查是否存在之前训练的模型，如果有则加载最优模型
    model_files = [f for f in os.listdir('.') if f.startswith('jsp_model_best_') and f.endswith('.pth')]
    if model_files:
        # 找到 makespan 最小的模型（文件名中数字最小的）
        best_model = min(model_files, key=lambda x: int(x.split('_')[3].split('.')[0]))
        print(f"Loading best model: {best_model}")
        ppo.load(best_model)
        print(f"Loaded model with makespan: {int(best_model.split('_')[3].split('.')[0])}")

    makespan_history = []
    
    # 保存最后一次环境用于可视化
    last_env = None

    for episode in range(n_episodes):
        # 创建环境
        env = JSP_Env(n_jobs, n_machines, proc_times, machine_assign, device)
        state = env.reset()

        episode_reward = 0
        done = False

        while not done:
            # 选择动作（仅选择工件）
            action, log_prob = ppo.select_action(state)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存储转移
            ppo.store_transition(state, action, reward, next_state, done, log_prob)

            state = next_state
            episode_reward += reward
        
        last_env = env

        # 记录结果
        makespan = env.get_makespan()
        makespan_history.append(makespan)

        # 定期更新策略
        if episode % 10 == 0 and episode > 0:
            ppo.update()

        if episode % 50 == 0:
            avg_makespan = np.mean(makespan_history[-50:])
            print(f"Episode {episode}, Makespan: {makespan}, Avg(50): {avg_makespan:.2f}")
    
    # 保存最终模型和最优模型
    final_makespan = makespan_history[-1]
    
    # 保存本次训练的模型到对应文件夹
    current_model_path = os.path.join(results_dir, f'policy_job.pth')
    ppo.save(current_model_path)
    print(f"Model saved to {current_model_path}")
    
    # 保存训练历史
    training_history_file = os.path.join(results_dir, 'training_history.npy')
    np.save(training_history_file, makespan_history)
    print(f"Training history saved to {training_history_file}")
    
    # 更新全局最优模型
    model_files = [f for f in os.listdir('.') if f.startswith('jsp_model_best_') and f.endswith('.pth')]
    if not model_files:
        # 如果还没有最优模型，直接保存当前模型为最优
        best_model_path = f'jsp_model_best_{int(final_makespan)}.pth'
        ppo.save(best_model_path)
        print(f"New best model saved to {best_model_path} (Makespan: {final_makespan})")
    else:
        # 找到当前最优模型的 makespan
        best_makespan = min([int(f.split('_')[3].split('.')[0]) for f in model_files])
        
        # 如果当前模型更优，则保存为新的最优模型
        if final_makespan < best_makespan:
            # 删除旧的最优模型
            for old_best in model_files:
                os.remove(old_best)
                print(f"Removed old best model: {old_best}")
            
            # 保存新的最优模型
            new_best_path = f'jsp_model_best_{int(final_makespan)}.pth'
            ppo.save(new_best_path)
            print(f"New best model saved to {new_best_path} (Makespan: {final_makespan})")
        else:
            print(f"Current model not better than best (Best: {best_makespan}, Current: {final_makespan})")
    
    # 可视化最后一次调度结果
    if last_env:
        print("Generating visualizations...")
        plot_train_graph(last_env.schedule, n_machines, n_jobs, os.path.join(results_dir, 'train_graph.png'))
        plot_gantt_chart(last_env.schedule, n_machines, n_jobs, os.path.join(results_dir, 'gantt_chart.png'))
        save_schedule_to_csv(last_env.schedule, os.path.join(results_dir, 'train_schedule.csv'))

    # 绘制学习曲线
    plt.figure()
    plt.plot(makespan_history)
    plt.xlabel('Episode')
    plt.ylabel('Makespan')
    plt.title(f'JSP Training Curve ({timestamp})')
    plt.grid(True, linestyle='--', alpha=0.7)
    learning_curve_file = os.path.join(results_dir, 'training_curve.png')
    plt.savefig(learning_curve_file)
    print(f"Learning curve saved to {learning_curve_file}")
    plt.close()


if __name__ == '__main__':
    train()