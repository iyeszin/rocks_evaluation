import matplotlib.pyplot as plt

def plot_mineral_analysis(analysis_history, rock_num, save_path=None):
    if not analysis_history:
        print("No data to plot")
        return
        
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(10, 6), dpi=300)
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    ax1 = fig.add_subplot(gs[0])
    measurements = range(1, len(analysis_history) + 1)
    
    true_minerals = [a['true_mineral'] for a in analysis_history]
    predicted_minerals = [a['mineral_prediction'] for a in analysis_history]
    
    ax1.scatter(measurements, true_minerals, label='Ground Truth', 
            marker='o', s=100, alpha=0.6)
    ax1.scatter(measurements, predicted_minerals, label='Predicted',
            marker='x', s=100, alpha=0.8)
    
    ax1.set_title(f'Mineral Predictions vs Ground Truth of Test Sample {rock_num}')
    ax1.set_xlabel('Measurement Number')
    ax1.set_ylabel('Mineral')
    ax1.legend(loc='upper left')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()