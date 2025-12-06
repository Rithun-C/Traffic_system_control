"""
Performance Graph Generator for Traffic Detection Video Analysis

Generates comprehensive performance graphs and saves them to experiments/<timestamp>/graphs/
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
from typing import Dict, List
import numpy as np

def create_experiments_directory(base_path: str = "experiments") -> str:
    """
    Create experiments directory with timestamp subdirectory and graphs folder.

    Args:
        base_path: Base directory for experiments

    Returns:
        str: Path to the graphs directory for this experiment
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_path, timestamp)
    graphs_dir = os.path.join(experiment_dir, "graphs")

    os.makedirs(graphs_dir, exist_ok=True)
    print(f"Created experiment directory: {experiment_dir}")
    return graphs_dir

def generate_vehicle_count_timeline(video_results: Dict, graphs_dir: str) -> str:
    """
    Generate timeline graph showing vehicle count over video duration.
    """
    if not video_results['success'] or not video_results['frame_results']:
        print("No frame results available for vehicle count timeline")
        return None

    frame_results = video_results['frame_results']

    # Extract data
    timestamps = [fr['timestamp'] for fr in frame_results]
    vehicle_counts = [fr['vehicle_count'] for fr in frame_results]

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, vehicle_counts, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7)
    plt.fill_between(timestamps, vehicle_counts, alpha=0.3, color='blue')

    # Add peak traffic annotation
    max_vehicles = max(vehicle_counts)
    peak_idx = vehicle_counts.index(max_vehicles)
    peak_time = timestamps[peak_idx]
    plt.annotate(f'Peak: {max_vehicles} vehicles\nat {peak_time:.1f}s',
                xy=(peak_time, max_vehicles),
                xytext=(peak_time + max(timestamps) * 0.1, max_vehicles),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    plt.title('Vehicle Count Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Number of Vehicles Detected', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    filename = os.path.join(graphs_dir, 'vehicle_count_timeline.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Generated vehicle count timeline: {filename}")
    return filename

def generate_performance_metrics_chart(video_results: Dict, graphs_dir: str) -> str:
    """
    Generate performance metrics visualization (processing times, FPS, etc.).
    """
    if not video_results['success']:
        print("No performance metrics available")
        return None

    metrics = video_results['performance_metrics']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Frame processing time distribution
    processing_times = metrics['frame_processing_times']
    ax1.hist(processing_times, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_title('Frame Processing Time Distribution', fontweight='bold')
    ax1.set_xlabel('Processing Time (seconds)')
    ax1.set_ylabel('Number of Frames')
    ax1.grid(True, alpha=0.3)

    # Add statistics
    avg_time = np.mean(processing_times)
    ax1.axvline(avg_time, color='red', linestyle='--', label=f'Avg: {avg_time:.3f}s')
    ax1.legend()

    # 2. Processing FPS over time
    frame_numbers = list(range(0, len(processing_times)))
    instantaneous_fps = [1.0 / pt if pt > 0 else 0 for pt in processing_times]
    ax2.plot(frame_numbers, instantaneous_fps, 'g-', alpha=0.7)
    ax2.set_title('Processing FPS Over Time', fontweight='bold')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Processing FPS')
    ax2.grid(True, alpha=0.3)

    # Add average line
    avg_fps = metrics['processing_fps']
    ax2.axhline(avg_fps, color='red', linestyle='--', label=f'Avg: {avg_fps:.1f} FPS')
    ax2.legend()

    # 3. Video vs Processing Speed Comparison
    video_info = video_results['video_info']
    categories = ['Video FPS', 'Processing FPS', 'Real-time Ratio']
    values = [video_info['fps'], avg_fps, avg_fps / video_info['fps'] if video_info['fps'] > 0 else 0]
    colors = ['blue', 'green', 'orange']

    bars = ax3.bar(categories, values, color=colors, alpha=0.7)
    ax3.set_title('Performance Comparison', fontweight='bold')
    ax3.set_ylabel('FPS / Ratio')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    # Add real-time performance indicator
    realtime_ratio = values[2]
    if realtime_ratio >= 1.0:
        performance_text = "✓ Real-time capable"
        text_color = 'green'
    else:
        performance_text = f"⚠ {realtime_ratio:.2f}x slower than real-time"
        text_color = 'red'

    ax3.text(0.5, 0.95, performance_text, transform=ax3.transAxes,
             ha='center', va='top', fontsize=12, color=text_color, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # 4. Summary statistics
    ax4.axis('off')
    stats_text = f"""
    Video Summary:
    • Duration: {metrics['duration']:.2f} seconds
    • Total Frames: {metrics['total_frames']:,}
    • Frames Processed: {metrics['frames_processed']:,}
    • Sample Rate: Every {metrics['sample_rate']} frame(s)

    Processing Performance:
    • Total Processing Time: {metrics['total_processing_time']:.2f}s
    • Average Frame Time: {metrics['avg_frame_processing_time']:.3f}s
    • Processing FPS: {metrics['processing_fps']:.1f}
    • Efficiency: {(metrics['processing_fps'] / video_info['fps'] * 100):.1f}% of real-time
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             va='top', ha='left', fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    # Save the plot
    filename = os.path.join(graphs_dir, 'performance_metrics.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Generated performance metrics chart: {filename}")
    return filename

def generate_vehicle_type_analysis(video_results: Dict, graphs_dir: str) -> str:
    """
    Generate vehicle type distribution and trends analysis.
    """
    if not video_results['success'] or not video_results['frame_results']:
        print("No frame results available for vehicle type analysis")
        return None

    frame_results = video_results['frame_results']

    # Aggregate vehicle type data
    vehicle_types = ['car', 'motorcycle', 'bus', 'truck']
    type_counts = {vtype: [] for vtype in vehicle_types}
    timestamps = []

    for fr in frame_results:
        timestamps.append(fr['timestamp'])
        summary = fr['vehicle_summary']
        for vtype in vehicle_types:
            type_counts[vtype].append(summary.get(vtype, 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Vehicle type timeline (stacked area chart)
    ax1.stackplot(timestamps, *type_counts.values(),
                 labels=vehicle_types, alpha=0.7,
                 colors=['blue', 'red', 'green', 'orange'])
    ax1.set_title('Vehicle Types Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Number of Vehicles')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Overall vehicle type distribution (pie chart)
    total_counts = {vtype: sum(counts) for vtype, counts in type_counts.items()}
    # Only include types that were detected
    detected_types = {k: v for k, v in total_counts.items() if v > 0}

    if detected_types:
        ax2.pie(detected_types.values(), labels=detected_types.keys(), autopct='%1.1f%%',
               colors=['blue', 'red', 'green', 'orange'][:len(detected_types)])
        ax2.set_title('Overall Vehicle Type Distribution', fontsize=14, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No vehicles detected', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Vehicle Type Distribution - No Data', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save the plot
    filename = os.path.join(graphs_dir, 'vehicle_type_analysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Generated vehicle type analysis: {filename}")
    return filename

def generate_traffic_signal_recommendations(video_results: Dict, graphs_dir: str) -> str:
    """
    Generate graph showing traffic signal timing recommendations over time.
    """
    if not video_results['success'] or not video_results['frame_results']:
        print("No frame results available for signal recommendations")
        return None

    frame_results = video_results['frame_results']

    # Extract signal timing data
    timestamps = [fr['timestamp'] for fr in frame_results]
    green_durations = [fr['suggested_timing']['green_duration'] for fr in frame_results]
    vehicle_counts = [fr['vehicle_count'] for fr in frame_results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 1. Green signal duration recommendations over time
    ax1.plot(timestamps, green_durations, 'g-', linewidth=2, label='Recommended Green Duration')
    ax1.fill_between(timestamps, green_durations, alpha=0.3, color='green')

    # Add horizontal lines for min/max durations
    ax1.axhline(30, color='red', linestyle='--', alpha=0.7, label='Minimum Duration (30s)')
    ax1.axhline(120, color='red', linestyle='--', alpha=0.7, label='Maximum Duration (120s)')

    ax1.set_title('Traffic Signal Green Duration Recommendations', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Green Duration (seconds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(20, 130)

    # 2. Vehicle count vs green duration correlation
    ax2.scatter(vehicle_counts, green_durations, alpha=0.6, color='blue', s=50)

    # Add trend line
    if len(vehicle_counts) > 1:
        z = np.polyfit(vehicle_counts, green_durations, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(vehicle_counts), max(vehicle_counts), 100)
        ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend Line')

    # Show the formula used
    formula_text = "Green Duration = max(30, min(120, vehicle_count × 5))"
    ax2.text(0.05, 0.95, formula_text, transform=ax2.transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
             fontsize=10, fontweight='bold')

    ax2.set_title('Vehicle Count vs Green Duration Correlation', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Vehicle Count')
    ax2.set_ylabel('Green Duration (seconds)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save the plot
    filename = os.path.join(graphs_dir, 'traffic_signal_recommendations.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Generated traffic signal recommendations: {filename}")
    return filename

def generate_all_performance_graphs(video_results: Dict, experiment_name: str = None) -> str:
    """
    Generate all performance graphs for video analysis results.

    Args:
        video_results: Results from detect_vehicles_in_video function
        experiment_name: Optional name for the experiment

    Returns:
        str: Path to the graphs directory containing all generated graphs
    """
    if not video_results['success']:
        print(f"Cannot generate graphs - video processing failed: {video_results.get('error', 'Unknown error')}")
        return None

    # Create experiment directory
    graphs_dir = create_experiments_directory()

    print(f"\nGenerating performance graphs for video analysis...")
    print(f"Experiment directory: {graphs_dir}")

    generated_files = []

    # Generate all graph types
    try:
        file1 = generate_vehicle_count_timeline(video_results, graphs_dir)
        if file1: generated_files.append(file1)

        file2 = generate_performance_metrics_chart(video_results, graphs_dir)
        if file2: generated_files.append(file2)

        file3 = generate_vehicle_type_analysis(video_results, graphs_dir)
        if file3: generated_files.append(file3)

        file4 = generate_traffic_signal_recommendations(video_results, graphs_dir)
        if file4: generated_files.append(file4)

        # Create a summary text file
        summary_file = os.path.join(graphs_dir, 'experiment_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Video Analysis Experiment Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment Name: {experiment_name or 'Unnamed'}\n\n")

            if video_results['video_info']:
                f.write(f"Video Information:\n")
                f.write(f"  Duration: {video_results['video_info']['duration']:.2f} seconds\n")
                f.write(f"  FPS: {video_results['video_info']['fps']:.2f}\n")
                f.write(f"  Total Frames: {video_results['video_info']['total_frames']}\n")
                f.write(f"  Sample Rate: {video_results['video_info']['sample_rate']}\n\n")

            if video_results['aggregate_stats']:
                stats = video_results['aggregate_stats']
                f.write(f"Analysis Results:\n")
                f.write(f"  Total Vehicles Detected: {stats['total_vehicle_detections']}\n")
                f.write(f"  Average Vehicles per Frame: {stats['avg_vehicles_per_frame']:.2f}\n")
                f.write(f"  Peak Traffic: {stats['max_vehicles_in_frame']} vehicles at {stats['peak_traffic_timestamp']:.1f}s\n")
                f.write(f"  Frames with Vehicles: {stats['frames_with_vehicles']}\n\n")

            if video_results['overall_suggested_timing']:
                timing = video_results['overall_suggested_timing']
                f.write(f"Overall Signal Timing Recommendation:\n")
                f.write(f"  Green Duration: {timing['green_duration']} seconds\n")
                f.write(f"  Red Duration: {timing['red_duration']} seconds\n")
                f.write(f"  Yellow Duration: {timing['yellow_duration']} seconds\n\n")

            f.write(f"Generated Graphs:\n")
            for i, filepath in enumerate(generated_files, 1):
                f.write(f"  {i}. {os.path.basename(filepath)}\n")

        print(f"\n✅ Successfully generated {len(generated_files)} performance graphs!")
        print(f"📁 All files saved in: {graphs_dir}")
        print(f"📄 Summary: {summary_file}")

        return graphs_dir

    except Exception as e:
        print(f"Error generating graphs: {e}")
        return None

if __name__ == "__main__":
    print("Graph Generator Module - Use generate_all_performance_graphs() function")