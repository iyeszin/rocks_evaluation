import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results(file_path='rock_classifications.csv'):
    # Load saved classifications
    results_df = pd.read_csv(file_path)
    
    # Ground truth list
    # Borderline is consider the rock type
    # Not the rock type is consider other
    ground_truths = [
        'granite', 'granite', 'other', 'granite', 'other', 
        'granite', 'granite', 'other', 'other', 'other',
        'sandstone', 'sandstone', 'other', 'sandstone', 'sandstone',
        'sandstone', 'other', 'sandstone', 'other', 'sandstone',
        'limestone', 'other', 'limestone', 'other', 'limestone',
        'limestone', 'limestone', 'other', 'limestone', 'limestone'
    ]
    
    # Create ground truth dataframe
    ground_truth_df = pd.DataFrame({
        'rock_num': range(1, 31),
        'ground_truth': ground_truths
    })
    
    # Merge and analyze
    final_df = pd.merge(results_df, ground_truth_df, on='rock_num')
    
    cm = confusion_matrix(final_df['ground_truth'], final_df['classification'])
    labels = ['Granite', 'Limestone', 'Sandstone', 'Other']
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8,6))  # Standard single-column width
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                annot_kws={'size': 12},  # Annotation font size
                )

    # Adjust font sizes
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    # Adjust tick label sizes
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()