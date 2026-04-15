import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATASET_PATH = 'C:/Users/adity/OneDrive/Desktop/OS Project/dataset'
RESULTS_PATH = 'C:/Users/adity/OneDrive/Desktop/OS Project/results'
FIGURES_PATH = 'C:/Users/adity/OneDrive/Desktop/OS Project/results/figures'
METRICS_PATH = 'C:/Users/adity/OneDrive/Desktop/OS Project/results/metrics'

# create output folders if they don't exist
os.makedirs(FIGURES_PATH, exist_ok=True)
os.makedirs(METRICS_PATH, exist_ok=True)

SCHEDULERS  = ['FCFS', 'SJF', 'RR', 'DDQN']
LOADS       = ['light', 'medium', 'heavy']
RUNS        = ['run1', 'run2', 'run3']
TASK_TYPES  = ['cpu', 'memory', 'io']

# colors matching paper style
COLORS = {
    'FCFS': '#E74C3C',
    'SJF':  '#3498DB',
    'RR':   '#2ECC71',
    'DDQN': '#9B59B6'
}

LOAD_COLORS = {
    'light':  '#82E0AA',
    'medium': '#F39C12',
    'heavy':  '#E74C3C'
}

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD ALL CSV FILES AND AVERAGE 3 RUNS
# ─────────────────────────────────────────────────────────────
def load_and_average():
    """
    Load all 12 conditions x 3 runs = 36 CSV files
    Average each condition across 3 runs
    Returns dict: results[scheduler][load] = averaged dataframe
    """
    results = {}

    for scheduler in SCHEDULERS:
        results[scheduler] = {}
        for load in LOADS:
            run_dfs = []
            for run in RUNS:
                filename = f'{scheduler}_{load}_{run}_metrics.csv'
                filepath = os.path.join(DATASET_PATH, filename)
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    run_dfs.append(df)
                else:
                    print(f"WARNING: Missing file: {filename}")

            if run_dfs:
                # average metrics across all runs for this condition
                combined = pd.concat(run_dfs, ignore_index=True)
                results[scheduler][load] = combined
                print(f"Loaded {scheduler} {load}: {len(combined)} total records across {len(run_dfs)} runs")
            else:
                print(f"ERROR: No data found for {scheduler} {load}")

    return results


# ─────────────────────────────────────────────────────────────
# STEP 2: CALCULATE SUMMARY METRICS
# ─────────────────────────────────────────────────────────────
def calculate_metrics(results):
    """
    Calculate the 3 paper metrics for each scheduler x load condition
    averaged across all 3 runs
    Returns summary dict and dataframe
    """
    summary = []

    for scheduler in SCHEDULERS:
        for load in LOADS:
            if load not in results[scheduler]:
                continue

            df = results[scheduler][load]

            avg_completion = df['completion_time'].mean()
            avg_response   = df['response_time'].mean()
            throughput     = len(df) / df['completion_time'].sum() * 1000

            # standard deviation across runs for error bars
            std_completion = df['completion_time'].std()
            std_response   = df['response_time'].std()

            summary.append({
                'scheduler':        scheduler,
                'load':             load,
                'avg_completion':   round(avg_completion, 2),
                'std_completion':   round(std_completion, 2),
                'throughput':       round(throughput, 3),
                'avg_response':     round(avg_response, 2),
                'std_response':     round(std_response, 2),
                'total_requests':   len(df)
            })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(METRICS_PATH, 'summary_metrics.csv'), index=False)
    print("\nSummary metrics saved.")
    return summary_df


# ─────────────────────────────────────────────────────────────
# STEP 3: REPRODUCE TABLE 1
# ─────────────────────────────────────────────────────────────
def print_table1(summary_df):
    """
    Reproduce the paper's Table 1 using averaged real measurements
    Shows overall performance across all load conditions
    """
    print("\n" + "="*75)
    print("TABLE 1: EXPERIMENTAL RESULTS (Averaged Across 3 Runs Per Condition)")
    print("="*75)
    print(f"{'Algorithm':<10} {'Avg Completion(ms)':<22} {'Throughput(tasks/s)':<22} {'Avg Response(ms)':<18}")
    print("-"*75)

    # paper's original values for comparison
    paper_values = {
        'FCFS': (350, 2.8, 200),
        'SJF':  (290, 3.1, 170),
        'RR':   (310, 3.0, 180),
        'DDQN': (250, 3.5, 150)
    }

    table1_data = []
    for scheduler in SCHEDULERS:
        sched_df = summary_df[summary_df['scheduler'] == scheduler]
        if sched_df.empty:
            continue

        # average across all load conditions for overall Table 1
        avg_completion = sched_df['avg_completion'].mean()
        avg_throughput = sched_df['throughput'].mean()
        avg_response   = sched_df['avg_response'].mean()

        print(f"{scheduler:<10} {avg_completion:<22.2f} {avg_throughput:<22.3f} {avg_response:<18.2f}")
        table1_data.append({
            'Algorithm':            scheduler,
            'Avg_Completion_ms':    round(avg_completion, 2),
            'Throughput_tasks_sec': round(avg_throughput, 3),
            'Avg_Response_ms':      round(avg_response, 2)
        })

    print("-"*75)
    print("\nPaper's Original Table 1 (Simulated):")
    print(f"{'Algorithm':<10} {'Avg Completion(ms)':<22} {'Throughput(tasks/s)':<22} {'Avg Response(ms)':<18}")
    print("-"*75)
    for scheduler, (comp, tp, resp) in paper_values.items():
        print(f"{scheduler:<10} {comp:<22} {tp:<22} {resp:<18}")
    print("="*75)

    # save table 1
    pd.DataFrame(table1_data).to_csv(
        os.path.join(METRICS_PATH, 'table1_results.csv'), index=False
    )
    return table1_data


# ─────────────────────────────────────────────────────────────
# STEP 4: REPRODUCE FIGURE 2 - Performance Under Different Loads
# ─────────────────────────────────────────────────────────────
def plot_figure2(summary_df):
    """
    Reproduce Figure 2: scheduler performance under light/medium/heavy load
    4 subplots: completion time, throughput, response time, scatter
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Figure 2: Scheduler Performance Under Different System Load Conditions\n(Real Windows OS Measurements, Averaged Across 3 Runs)',
        fontsize=13, fontweight='bold', y=1.02
    )

    x = np.arange(len(LOADS))
    width = 0.2

    # ── Subplot 1: Average Task Completion Time ──────────────
    ax1 = axes[0, 0]
    for i, scheduler in enumerate(SCHEDULERS):
        sched_df = summary_df[summary_df['scheduler'] == scheduler]
        values = [sched_df[sched_df['load'] == load]['avg_completion'].values[0]
                  if not sched_df[sched_df['load'] == load].empty else 0
                  for load in LOADS]
        errors = [sched_df[sched_df['load'] == load]['std_completion'].values[0]
                  if not sched_df[sched_df['load'] == load].empty else 0
                  for load in LOADS]
        bars = ax1.bar(x + i*width, values, width, label=scheduler,
                       color=COLORS[scheduler], alpha=0.85,
                       yerr=errors, capsize=3, error_kw={'linewidth': 1})

    ax1.set_xlabel('System Load', fontsize=11)
    ax1.set_ylabel('Avg Completion Time (ms)', fontsize=11)
    ax1.set_title('Task Completion Time', fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(['Light', 'Medium', 'Heavy'])
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # ── Subplot 2: System Throughput ─────────────────────────
    ax2 = axes[0, 1]
    for i, scheduler in enumerate(SCHEDULERS):
        sched_df = summary_df[summary_df['scheduler'] == scheduler]
        values = [sched_df[sched_df['load'] == load]['throughput'].values[0]
                  if not sched_df[sched_df['load'] == load].empty else 0
                  for load in LOADS]
        ax2.plot(LOADS, values, marker='o', linewidth=2,
                 label=scheduler, color=COLORS[scheduler], markersize=8)

    ax2.set_xlabel('System Load', fontsize=11)
    ax2.set_ylabel('Throughput (tasks/sec)', fontsize=11)
    ax2.set_title('System Throughput', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # ── Subplot 3: Average Response Time ─────────────────────
    ax3 = axes[1, 0]
    for i, scheduler in enumerate(SCHEDULERS):
        sched_df = summary_df[summary_df['scheduler'] == scheduler]
        values = [sched_df[sched_df['load'] == load]['avg_response'].values[0]
                  if not sched_df[sched_df['load'] == load].empty else 0
                  for load in LOADS]
        errors = [sched_df[sched_df['load'] == load]['std_response'].values[0]
                  if not sched_df[sched_df['load'] == load].empty else 0
                  for load in LOADS]
        ax3.bar(x + i*width, values, width, label=scheduler,
                color=COLORS[scheduler], alpha=0.85,
                yerr=errors, capsize=3, error_kw={'linewidth': 1})

    ax3.set_xlabel('System Load', fontsize=11)
    ax3.set_ylabel('Avg Response Time (ms)', fontsize=11)
    ax3.set_title('Response Time', fontweight='bold')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(['Light', 'Medium', 'Heavy'])
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # ── Subplot 4: Scatter - Completion Time vs Throughput ───
    ax4 = axes[1, 1]
    for scheduler in SCHEDULERS:
        sched_df = summary_df[summary_df['scheduler'] == scheduler]
        ax4.scatter(
            sched_df['avg_completion'],
            sched_df['throughput'],
            color=COLORS[scheduler], s=120, label=scheduler,
            zorder=5, edgecolors='white', linewidth=1.5
        )
        # connect the 3 load points with a line
        ax4.plot(
            sched_df['avg_completion'].values,
            sched_df['throughput'].values,
            color=COLORS[scheduler], alpha=0.4, linewidth=1
        )

    ax4.set_xlabel('Avg Completion Time (ms)', fontsize=11)
    ax4.set_ylabel('Throughput (tasks/sec)', fontsize=11)
    ax4.set_title('Completion Time vs Throughput', fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_PATH, 'figure2_load_performance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure 2 saved: {path}")


# ─────────────────────────────────────────────────────────────
# STEP 5: REPRODUCE FIGURE 3 - Performance By Task Type
# ─────────────────────────────────────────────────────────────
def plot_figure3(results):
    """
    Reproduce Figure 3: scheduler performance per task type
    CPU-bound vs Memory-bound vs IO-bound
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Figure 3: Scheduling Performance Under Different Task Types\n(Real Windows OS Measurements, Averaged Across 3 Runs)',
        fontsize=13, fontweight='bold', y=1.02
    )

    # aggregate all load conditions for task type analysis
    task_metrics = {scheduler: {task: {'completion': [], 'cpu': [], 'memory': []}
                                 for task in TASK_TYPES}
                    for scheduler in SCHEDULERS}

    for scheduler in SCHEDULERS:
        for load in LOADS:
            if load not in results[scheduler]:
                continue
            df = results[scheduler][load]
            for task in TASK_TYPES:
                task_df = df[df['task_type'] == task]
                if not task_df.empty:
                    task_metrics[scheduler][task]['completion'].append(
                        task_df['completion_time'].mean()
                    )
                    task_metrics[scheduler][task]['cpu'].append(
                        task_df['system_cpu_load'].mean()
                    )
                    task_metrics[scheduler][task]['memory'].append(
                        task_df['memory_available'].mean()
                    )

    x = np.arange(len(TASK_TYPES))
    width = 0.2
    task_labels = ['CPU-bound', 'Memory-bound', 'I/O-bound']

    # ── Subplot 1: Task Completion Time By Type ──────────────
    ax1 = axes[0, 0]
    for i, scheduler in enumerate(SCHEDULERS):
        values = [np.mean(task_metrics[scheduler][task]['completion'])
                  if task_metrics[scheduler][task]['completion'] else 0
                  for task in TASK_TYPES]
        ax1.bar(x + i*width, values, width, label=scheduler,
                color=COLORS[scheduler], alpha=0.85)

    ax1.set_xlabel('Task Type', fontsize=11)
    ax1.set_ylabel('Avg Completion Time (ms)', fontsize=11)
    ax1.set_title('Task Completion Time by Type', fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(task_labels)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # ── Subplot 2: CPU Utilization By Task Type ──────────────
    ax2 = axes[0, 1]
    for i, scheduler in enumerate(SCHEDULERS):
        values = [np.mean(task_metrics[scheduler][task]['cpu'])
                  if task_metrics[scheduler][task]['cpu'] else 0
                  for task in TASK_TYPES]
        ax2.bar(x + i*width, values, width, label=scheduler,
                color=COLORS[scheduler], alpha=0.85)

    ax2.set_xlabel('Task Type', fontsize=11)
    ax2.set_ylabel('CPU Utilization (%)', fontsize=11)
    ax2.set_title('CPU Utilization by Task Type', fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(task_labels)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # ── Subplot 3: Memory Utilization By Task Type ───────────
    ax3 = axes[1, 0]
    for i, scheduler in enumerate(SCHEDULERS):
        values = [np.mean(task_metrics[scheduler][task]['memory'])
                  if task_metrics[scheduler][task]['memory'] else 0
                  for task in TASK_TYPES]
        # invert since memory_available decreases as usage increases
        ax3.bar(x + i*width, values, width, label=scheduler,
                color=COLORS[scheduler], alpha=0.85)

    ax3.set_xlabel('Task Type', fontsize=11)
    ax3.set_ylabel('Memory Available (GB)', fontsize=11)
    ax3.set_title('Memory Utilization by Task Type', fontweight='bold')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(task_labels)
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # ── Subplot 4: Throughput By Task Type ───────────────────
    ax4 = axes[1, 1]
    for i, scheduler in enumerate(SCHEDULERS):
        values = []
        for task in TASK_TYPES:
            completions = task_metrics[scheduler][task]['completion']
            if completions:
                avg = np.mean(completions)
                tp = 1000 / avg if avg > 0 else 0
                values.append(tp)
            else:
                values.append(0)
        ax4.bar(x + i*width, values, width, label=scheduler,
                color=COLORS[scheduler], alpha=0.85)

    ax4.set_xlabel('Task Type', fontsize=11)
    ax4.set_ylabel('Throughput (tasks/sec)', fontsize=11)
    ax4.set_title('Throughput by Task Type', fontweight='bold')
    ax4.set_xticks(x + width * 1.5)
    ax4.set_xticklabels(task_labels)
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_PATH, 'figure3_task_type_performance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure 3 saved: {path}")


# ─────────────────────────────────────────────────────────────
# STEP 6: REPRODUCE FIGURE 4 - DDQN Loss Curve
# ─────────────────────────────────────────────────────────────
def plot_figure4():
    """
    Reproduce Figure 4: DDQN loss function over training epochs
    Loads loss history from saved model file
    """
    model_path = os.path.join(DATASET_PATH, 'ddqn_model.pth')

    if not os.path.exists(model_path):
        print("WARNING: ddqn_model.pth not found. Skipping Figure 4.")
        return

    import torch
    checkpoint = torch.load(model_path, map_location='cpu')
    loss_history = checkpoint.get('loss_history', [])

    if not loss_history:
        print("WARNING: No loss history found in model. Skipping Figure 4.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle(
        'Figure 4: DDQN Loss Function Over Training Epochs\n(Real Windows OS - Averaged Across 3 Runs)',
        fontsize=13, fontweight='bold'
    )

    epochs = range(len(loss_history))
    ax.plot(epochs, loss_history, color='#9B59B6', linewidth=1, alpha=0.6, label='Loss per step')

    # smooth rolling average
    if len(loss_history) > 20:
        window = max(10, len(loss_history) // 50)
        rolling = pd.Series(loss_history).rolling(window=window, center=True).mean()
        ax.plot(epochs, rolling, color='#1F3864', linewidth=2.5, label=f'Rolling avg (window={window})')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss vs Training Step', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # annotate convergence
    if len(loss_history) > 10:
        final_loss = np.mean(loss_history[-10:])
        ax.axhline(y=final_loss, color='red', linestyle='--', alpha=0.5,
                   label=f'Converged at ~{final_loss:.4f}')
        ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIGURES_PATH, 'figure4_loss_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure 4 saved: {path}")


# ─────────────────────────────────────────────────────────────
# STEP 7: COMPARISON TABLE - Our Results vs Paper
# ─────────────────────────────────────────────────────────────
def plot_comparison(table1_data):
    """
    Side by side bar chart comparing our real results vs paper's simulated results
    """
    paper = {
        'FCFS': {'completion': 350, 'throughput': 2.8, 'response': 200},
        'SJF':  {'completion': 290, 'throughput': 3.1, 'response': 170},
        'RR':   {'completion': 310, 'throughput': 3.0, 'response': 180},
        'DDQN': {'completion': 250, 'throughput': 3.5, 'response': 150}
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        'Our Real Results vs Paper Simulated Results\n(Lower is better for completion and response time, Higher is better for throughput)',
        fontsize=12, fontweight='bold', y=1.02
    )

    x = np.arange(len(SCHEDULERS))
    width = 0.35

    our_data = {row['Algorithm']: row for row in table1_data}

    # ── Completion Time ──────────────────────────────────────
    ax = axes[0]
    paper_vals = [paper[s]['completion'] for s in SCHEDULERS]
    our_vals   = [our_data[s]['Avg_Completion_ms'] if s in our_data else 0 for s in SCHEDULERS]
    ax.bar(x - width/2, paper_vals, width, label='Paper (Simulated)', color='#BDC3C7', alpha=0.9)
    ax.bar(x + width/2, our_vals,   width, label='Ours (Real)',       color='#2E75B6', alpha=0.9)
    ax.set_xlabel('Scheduler', fontsize=11)
    ax.set_ylabel('Avg Completion Time (ms)', fontsize=11)
    ax.set_title('Completion Time\n(lower is better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(SCHEDULERS)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # ── Throughput ───────────────────────────────────────────
    ax = axes[1]
    paper_vals = [paper[s]['throughput'] for s in SCHEDULERS]
    our_vals   = [our_data[s]['Throughput_tasks_sec'] if s in our_data else 0 for s in SCHEDULERS]
    ax.bar(x - width/2, paper_vals, width, label='Paper (Simulated)', color='#BDC3C7', alpha=0.9)
    ax.bar(x + width/2, our_vals,   width, label='Ours (Real)',       color='#27AE60', alpha=0.9)
    ax.set_xlabel('Scheduler', fontsize=11)
    ax.set_ylabel('Throughput (tasks/sec)', fontsize=11)
    ax.set_title('System Throughput\n(higher is better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(SCHEDULERS)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # ── Response Time ────────────────────────────────────────
    ax = axes[2]
    paper_vals = [paper[s]['response'] for s in SCHEDULERS]
    our_vals   = [our_data[s]['Avg_Response_ms'] if s in our_data else 0 for s in SCHEDULERS]
    ax.bar(x - width/2, paper_vals, width, label='Paper (Simulated)', color='#BDC3C7', alpha=0.9)
    ax.bar(x + width/2, our_vals,   width, label='Ours (Real)',       color='#E74C3C', alpha=0.9)
    ax.set_xlabel('Scheduler', fontsize=11)
    ax.set_ylabel('Avg Response Time (ms)', fontsize=11)
    ax.set_title('Response Time\n(lower is better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(SCHEDULERS)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_PATH, 'figure_comparison_paper_vs_ours.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison figure saved: {path}")


# ─────────────────────────────────────────────────────────────
# STEP 8: DDQN IMPROVEMENT OVER BASELINES
# ─────────────────────────────────────────────────────────────
def plot_ddqn_improvement(table1_data):
    """
    Show how much DDQN improves over each baseline scheduler
    """
    our_data = {row['Algorithm']: row for row in table1_data}

    if 'DDQN' not in our_data:
        print("WARNING: No DDQN data found. Skipping improvement plot.")
        return

    ddqn_completion = our_data['DDQN']['Avg_Completion_ms']
    ddqn_throughput = our_data['DDQN']['Throughput_tasks_sec']
    ddqn_response   = our_data['DDQN']['Avg_Response_ms']

    baselines = ['FCFS', 'SJF', 'RR']
    improvements_completion = []
    improvements_throughput = []
    improvements_response   = []

    for b in baselines:
        if b in our_data:
            comp_imp = ((our_data[b]['Avg_Completion_ms'] - ddqn_completion)
                        / our_data[b]['Avg_Completion_ms'] * 100)
            tp_imp   = ((ddqn_throughput - our_data[b]['Throughput_tasks_sec'])
                        / our_data[b]['Throughput_tasks_sec'] * 100)
            resp_imp = ((our_data[b]['Avg_Response_ms'] - ddqn_response)
                        / our_data[b]['Avg_Response_ms'] * 100)
        else:
            comp_imp = tp_imp = resp_imp = 0

        improvements_completion.append(comp_imp)
        improvements_throughput.append(tp_imp)
        improvements_response.append(resp_imp)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        'DDQN Improvement Over Baseline Schedulers (%)\n(Real Windows OS Measurements)',
        fontsize=13, fontweight='bold'
    )

    x = np.arange(len(baselines))
    width = 0.25
    ax.bar(x - width, improvements_completion, width, label='Completion Time Reduction (%)',
           color='#2E75B6', alpha=0.85)
    ax.bar(x,         improvements_throughput, width, label='Throughput Increase (%)',
           color='#27AE60', alpha=0.85)
    ax.bar(x + width, improvements_response,   width, label='Response Time Reduction (%)',
           color='#E74C3C', alpha=0.85)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Baseline Scheduler', fontsize=11)
    ax.set_ylabel('Improvement (%)', fontsize=11)
    ax.set_title('DDQN vs Baselines: Percentage Improvement', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(baselines)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_PATH, 'figure_ddqn_improvement.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"DDQN improvement figure saved: {path}")


# ─────────────────────────────────────────────────────────────
# MAIN: RUN ALL ANALYSIS
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("="*60)
    print("PHASE 9: ANALYSIS AND FIGURE GENERATION")
    print("="*60)

    print("\nStep 1: Loading all CSV files (3 runs per condition)...")
    results = load_and_average()

    print("\nStep 2: Calculating summary metrics...")
    summary_df = calculate_metrics(results)

    print("\nStep 3: Reproducing Table 1...")
    table1_data = print_table1(summary_df)

    print("\nStep 4: Reproducing Figure 2 (Load performance)...")
    plot_figure2(summary_df)

    print("\nStep 5: Reproducing Figure 3 (Task type performance)...")
    plot_figure3(results)

    print("\nStep 6: Reproducing Figure 4 (DDQN loss curve)...")
    plot_figure4()

    print("\nStep 7: Generating comparison chart (Ours vs Paper)...")
    plot_comparison(table1_data)

    print("\nStep 8: Generating DDQN improvement chart...")
    plot_ddqn_improvement(table1_data)

    print("\n" + "="*60)
    print("ALL ANALYSIS COMPLETE")
    print(f"Figures saved to: {FIGURES_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print("="*60)
    print("\nFiles generated:")
    print("  figures/figure2_load_performance.png")
    print("  figures/figure3_task_type_performance.png")
    print("  figures/figure4_loss_curve.png")
    print("  figures/figure_comparison_paper_vs_ours.png")
    print("  figures/figure_ddqn_improvement.png")
    print("  metrics/summary_metrics.csv")
    print("  metrics/table1_results.csv")