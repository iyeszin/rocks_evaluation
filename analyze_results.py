import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results():
    # Load saved classifications
    results_df = pd.read_csv('rock_classifications.csv')
    
    # Ground truth list
    # Borderline is consider the rock type
    # Not the rock type is consider other
    ground_truths = [
        'granite', 'granite', 'granite', 'granite', 'granite', 
        'granite', 'granite', 'granite', 'other', 'other',
        'sandstone', 'sandstone', 'sandstone', 'sandstone', 'sandstone',
        'sandstone', 'other', 'sandstone', 'sandstone', 'sandstone',
        'limestone', 'other', 'limestone', 'other', 'limestone',
        'limestone', 'limestone', 'limestone', 'limestone', 'limestone'
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
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, 
            yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print("\nClassification Report:")
    print(classification_report(final_df['ground_truth'], final_df['classification']))


    final_df.to_csv('rock_classification_analysis.csv', index=False)