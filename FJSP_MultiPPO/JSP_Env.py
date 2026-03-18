"""
JSP Environment for Train Timetabling Problem
"""
import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical


class JSP_Env:
    def __init__(self, n_jobs, n_machines, processing_time_matrix,
                 job_op_sequences, device='cuda'):
        """
        Args:
            n_jobs: 列车数量（工件数）
            n_machines: 区间/车站数量（机器数）
            processing_time_matrix: [n_jobs, max_ops_per_job] 每个工序的加工时间
            job_op_sequences: [n_jobs, max_ops_per_job] 每个工序对应的机器ID（固定分配）
            device: 'cuda' or 'cpu'
        """
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.device = device

        # JSP参数：每个工序有固定的机器和加工时间
        self.processing_time = processing_time_matrix  # [n_jobs, max_ops]
        self.op_machine_assign = job_op_sequences  # [n_jobs, max_ops] 固定机器分配

        self.max_ops = processing_time_matrix.shape[1]
        
        # 追踪间隔参数
        self.min_headway_normal = 3  # 普通列车间隔 3min
        self.min_headway_maintenance = 120 # 天窗列车间隔 120min
        self.maintenance_job_id = 4  # 假设第5列车（索引4）是天窗
        
        # 状态变量
        self.reset()

    def reset(self):
        """重置环境状态"""
        # 每个工件的当前工序索引
        self.current_op_idx = np.zeros(self.n_jobs, dtype=int)

        # 机器时间状态
        self.machine_available_time = np.zeros(self.n_machines)
        
        # 记录每台机器上一次的状态
        self.machine_last_job = -np.ones(self.n_machines, dtype=int)
        self.machine_last_leave_time = np.zeros(self.n_machines) # Exit time
        self.machine_last_enter_time = np.zeros(self.n_machines) # Enter time

        # 工件完成时间
        self.job_ready_time = np.zeros(self.n_jobs)

        # 完成标记
        self.completed_ops = np.zeros((self.n_jobs, self.max_ops), dtype=bool)
        self.completed_jobs = np.zeros(self.n_jobs, dtype=bool)

        # 调度结果记录
        self.schedule = {}  # (job, op) -> (machine, start, end)

        return self.get_state()

    def get_state(self):
        """
        获取当前状态
        返回可直接输入神经网络的状态特征
        """
        # 就绪工序特征 [n_jobs, 4]
        ready_ops_features = []

        for job_id in range(self.n_jobs):
            if self.completed_jobs[job_id]:
                # 工件已完成
                ready_ops_features.append([0, 0, 0, 0])
            else:
                op_idx = self.current_op_idx[job_id]
                machine_id = self.op_machine_assign[job_id, op_idx]
                proc_time = self.processing_time[job_id, op_idx]

                # 特征：工件ID, 工序ID, 机器ID, 加工时间
                ready_ops_features.append([
                    job_id / self.n_jobs,  # 归一化工件ID
                    op_idx / self.max_ops,  # 归一化工序序号
                    machine_id / self.n_machines,  # 归一化机器ID
                    proc_time / 100.0  # 归一化加工时间（假设最大100）
                ])

        # 机器状态特征 [n_machines, 2]
        machine_features = []
        for m in range(self.n_machines):
            machine_features.append([
                self.machine_available_time[m] / 1000.0,  # 归一化可用时间
                0  # 预留特征
            ])

        return {
            'ready_ops': torch.FloatTensor(ready_ops_features).to(self.device),
            'machines': torch.FloatTensor(machine_features).to(self.device),
            'mask': self.get_action_mask()
        }

    def get_action_mask(self):
        """
        获取动作掩码：标记哪些工件的就绪工序可以选择
        Returns: [n_jobs] 的bool mask，True表示该工件的就绪工序可选
        """
        mask = np.zeros(self.n_jobs, dtype=bool)

        for job_id in range(self.n_jobs):
            if not self.completed_jobs[job_id]:
                op_idx = self.current_op_idx[job_id]
                machine_id = self.op_machine_assign[job_id, op_idx]

                # JSP中只要机器空闲就可以选择（简化约束）
                # 实际运行图可能需要考虑区间占用冲突，这里简化处理
                mask[job_id] = True

        return torch.BoolTensor(mask).to(self.device)

    def step(self, job_id):
        """
        执行动作：为指定工件调度其当前就绪工序

        Args:
            job_id: 选择的工件ID（0到n_jobs-1）

        Returns:
            next_state, reward, done, info
        """
        op_idx = self.current_op_idx[job_id]
        machine_id = self.op_machine_assign[job_id, op_idx]
        proc_time = self.processing_time[job_id, op_idx]

        # ------------------------------------------------------------------
        # 1. 确定机器可用时间（Machine Availability）
        # ------------------------------------------------------------------
        # 在标准JSP中，机器可用时间是上一工序完成时间。
        # 但为了模拟“追踪间隔”，我们需要区分：
        #   (a) 物理占用释放时间（Physically Free Time）：列车离开区间/车站的时间
        #   (b) 下一列车允许进入时间（Next Train Entry Time）：
        #       - 如果上一列车是普通列车：Entry Time >= 上一列车Entry Time + 3 min
        #       - 如果上一列车是天窗（T4）：Entry Time >= 上一列车Exit Time + 120 min (严禁进入 -> 独占)
        #       - 如果当前列车是天窗（T4）：Entry Time >= 上一列车Exit Time (需要清空区间)
        
        # 我们使用 self.machine_available_time 来存储 "允许下一列车进入的最早时间"
        # 同时我们需要记录 "上一列车实际离开时间" (physical release time)
        
        # 获取当前机器的状态记录
        last_job = self.machine_last_job[machine_id]
        last_leave_time = self.machine_last_leave_time[machine_id] # 这里存储的是Exit Time (End Time)
        last_enter_time = self.machine_last_enter_time[machine_id] # 新增：记录上一列车Enter Time (Start Time)
        
        # 计算基于机器占用的最早开始时间
        machine_ready_time = 0
        
        if last_job != -1:
            # 判断上一列车类型
            is_last_maintenance = (last_job == self.maintenance_job_id)
            # 判断当前列车类型
            is_curr_maintenance = (job_id == self.maintenance_job_id)
            
            if is_curr_maintenance:
                # 当前是天窗：必须等上一列车完全离开，且可能需要额外确认（这里假设离开即可）
                # 严禁任何列车进入 -> 天窗需要独占
                machine_ready_time = last_leave_time
            elif is_last_maintenance:
                # 上一列车是天窗：当前列车必须等天窗结束后 120 min 才能进入
                machine_ready_time = last_leave_time + self.min_headway_maintenance
            else:
                # 上一列车是普通列车，当前也是普通列车
                # 追踪间隔 3 min (相对于上一列车进入时间)
                # 但同时也必须等上一列车离开吗？
                # 如果是区间（单线闭塞），通常必须等上一列车离开（或者分区块）。
                # 题目说 "列车1、2、3、4的追踪间隔为3min"。
                # 如果区间运行时间是15min，间隔3min意味着区间内可以有多列车（移动闭塞或多分区）。
                # 因此，我们允许 "Entry Time >= Last Entry Time + 3"
                # 且 "Entry Time >= Last Leave Time" (如果单闭塞) OR 忽略Leave Time constraint (如果多闭塞)
                # 鉴于题目强调3min间隔，这通常指发车间隔。我们假设区间容量足够（虚拟分区）。
                # 所以主要约束是 headway。
                
                # 但是，为了防止物理碰撞（后车追前车），如果后车比前车快，或者出站受阻，需要考虑。
                # 这里所有普通列车速度相同（120km/h），所以只要满足进入间隔，区间内就不会追尾。
                machine_ready_time = last_enter_time + self.min_headway_normal
                
                # 修正：虽然间隔满足，但如果上一列车还没走完，JSP模型默认是冲突的。
                # 为了在JSP框架下实现这一点，我们认为机器的 "availability" 是虚拟的 token availability.
                # 但 vis_utils 画图时可能会画出重叠。
                # 如果用户接受区间内多列车，则允许重叠。
                pass

        # 综合考虑工件就绪时间
        start_time = max(machine_ready_time, self.job_ready_time[job_id])
        end_time = start_time + proc_time

        # 更新状态
        self.schedule[(job_id, op_idx)] = (machine_id, start_time, end_time)
        
        # 更新机器状态
        self.machine_last_job[machine_id] = job_id
        self.machine_last_leave_time[machine_id] = end_time
        self.machine_last_enter_time[machine_id] = start_time
        
        # 更新机器可用时间（供下一次step使用）
        # 这里其实不需要显式维护 machine_available_time 数组作为唯一标准，
        # 因为我们已经通过 last_job 等变量动态计算了。
        # 但为了兼容旧代码的 reset/get_state，我们还是更新它。
        # 注意：get_state 中的 machine available time 特征可能需要调整，但在RL中让它自己学吧。
        self.machine_available_time[machine_id] = end_time # 仅作为参考，实际逻辑已在上面重写

        self.job_ready_time[job_id] = end_time
        self.completed_ops[job_id, op_idx] = True

        # 移动到下一工序
        self.current_op_idx[job_id] += 1
        if self.current_op_idx[job_id] >= self.max_ops or \
                self.processing_time[job_id, self.current_op_idx[job_id]] == 0 or \
                self.op_machine_assign[job_id, self.current_op_idx[job_id]] == -1:
            self.completed_jobs[job_id] = True

        # 计算奖励
        reward = self._compute_reward()
        done = np.all(self.completed_jobs)

        return self.get_state(), reward, done, {
            'makespan': max(self.job_ready_time), # 使用所有列车完成时间的最大值更准确
            'scheduled': (job_id, op_idx, machine_id, start_time, end_time)
        }

    def _compute_reward(self):
        """
        奖励函数：鼓励减小makespan
        可以使用多种奖励设计：
        1. 每步奖励：-当前最大完成时间增量
        2. 稀疏奖励：只有结束时给 -makespan
        3. 基于空闲时间的奖励
        """
        # 简单版本：结束时的负makespan，中间步骤奖励为0
        if np.all(self.completed_jobs):
            return -max(self.machine_available_time) / 100.0  # 归一化
        return 0.0

    def get_makespan(self):
        """获取当前makespan"""
        return max(self.machine_available_time)