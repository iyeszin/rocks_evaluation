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
       
       if 'accuracy_rule' in result['rock_analysis']:
           acc = result['rock_analysis']['accuracy_rule']
           f.write(f"Accuracy: {acc['accuracy']:.1%}\n")
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