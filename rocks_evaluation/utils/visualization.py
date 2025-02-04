import matplotlib.pyplot as plt
import numpy as np

def plot_mineral_analysis(analysis_history, rock_num, save_path=None):
    if not analysis_history:
        print("No data to plot")
        return

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15, 10), dpi=300)
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1])

    # Top plot: Mineral Predictions
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

    # Middle plot: Mineral Ratios
    ax2 = fig.add_subplot(gs[1])
    
    # Get mineral counts at each step
    mineral_ratios = []
    for i in range(len(analysis_history)):
        if 'rock_analysis' in analysis_history[i] and 'mineral_counts' in analysis_history[i]['rock_analysis']:
            counts = analysis_history[i]['rock_analysis']['mineral_counts']
            total = sum(counts.values()) if counts else 1
            ratios = {
                'Quartz': counts.get('quartz', 0) / total if total else 0,
                'Feldspars': counts.get('feldspars', 0) / total if total else 0,
                'Micas': counts.get('micas', 0) / total if total else 0
            }
            mineral_ratios.append(ratios)

    if mineral_ratios:
        measurements = range(1, len(mineral_ratios) + 1)
        for mineral in ['Quartz', 'Feldspars', 'Micas']:
            ratios = [r[mineral] for r in mineral_ratios]
            ax2.plot(measurements, ratios, label=f'{mineral} Ratio', marker='o')

    ax2.set_xlabel('Measurement Number')
    ax2.set_ylabel('Mineral Ratio')
    ax2.set_title('Mineral Ratios Evolution')
    ax2.legend()
    ax2.grid(True)

    # Bottom plot: Rock Type Weights
    ax3 = fig.add_subplot(gs[2])
    
    # Get weights at each step
    rock_weights = []
    for entry in analysis_history:
        if 'rock_analysis' in entry and 'weights' in entry['rock_analysis']:
            rock_weights.append(entry['rock_analysis']['weights'])

    if rock_weights:
        measurements = range(1, len(rock_weights) + 1)
        for rock_type in ['granite', 'sandstone', 'limestone']:
            weights = [w.get(rock_type, 0) for w in rock_weights]
            ax3.plot(measurements, weights, label=f'{rock_type.capitalize()} Weight', marker='o')
            
        # Add confidence threshold line
        ax3.axhline(y=0.7, color='r', linestyle='--', label='Confidence Threshold')

    ax3.set_xlabel('Measurement Number')
    ax3.set_ylabel('Weight')
    ax3.set_title('Rock Type Weights Evolution')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def save_analysis(result, filename="rock_analysis_results.txt"):
    """
    Save rock analysis results to a text file
    Args:
        result (dict): Analysis result dictionary
        filename (str): Name of output file
    """
    with open(filename, 'w') as f:
        # Analysis Results
        f.write("\nAnalysis Results:\n")
        f.write(f"Classification: {result['rock_analysis']['classification']}\n")
        
        # Add weights information
        if 'weights' in result['rock_analysis']:
            f.write("\nRock Type Weights:\n")
            for rock_type, weight in result['rock_analysis']['weights'].items():
                f.write(f"- {rock_type.capitalize()}: {weight:.3f}\n")
        
        # Add mineral counts
        if 'mineral_counts' in result['rock_analysis']:
            f.write("\nMineral Counts:\n")
            for mineral, count in result['rock_analysis']['mineral_counts'].items():
                f.write(f"- {mineral.capitalize()}: {count}\n")
        
        # Original accuracy information
        if 'accuracy_rule' in result['rock_analysis']:
            acc = result['rock_analysis']['accuracy_rule']
            f.write(f"\nAccuracy: {acc['accuracy']:.1%}\n")
            if 'unknown_count' in acc:
                f.write(f"Unknown predictions: {acc['unknown_count']}\n")
        
        # Mineral Assemblage
        if 'assemblage_rules' in result['rock_analysis']:
            f.write("\nMineral Assemblage:\n")
            for rule, satisfied in result['rock_analysis']['assemblage_rules']['details'].items():
                f.write(f"- {rule}: {'✓' if satisfied else '✗'}\n")
        
        # Uncertainty Metrics
        if 'uncertainty' in result:
            f.write("\nUncertainty Metrics:\n")
            f.write(f"Entropy: {result['uncertainty']['entropy']:.4f}\n")
            f.write(f"Variance: {result['uncertainty']['variance']:.4f}\n")