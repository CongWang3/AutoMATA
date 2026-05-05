import sys

import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from train.pseudo import load_model, predict, set_random_seed, ModelEvaluator, visualize_results, print_evaluation_report

import torch

import numpy as np

import argparse

import os

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

def load_and_predict(model_path, scaler_path, new_data, device, args=None):
    """
    Load model and predict new data
    """
    print("Loading model and preprocessor...")
    model, scaler = load_model(model_path, scaler_path, device)
    print(f"Model loaded, input dimension: {model.input_dim}")
    print(f"Number of classes: {model.num_classes}")
    print(f"New data shape: {new_data.shape}")
    # Check data dimension
    if new_data.shape[1] != model.input_dim:
        raise ValueError(f"Data dimension mismatch! Model expects {model.input_dim} dimensions, but input data has {new_data.shape[1]} dimensions")
    # Predict
    print("Predicting...")
    predictions, probabilities = predict(model, scaler, new_data, device)
    # Create class names (based on number of classes)
    class_names = [f'{i}' for i in range(model.num_classes)]
    predicted_labels = [class_names[pred] for pred in predictions]
    print("Prediction completed!")
    return {
        'original_data': new_data,
        'predictions': predictions,
        'predicted_labels': predicted_labels,
        'probabilities': probabilities,
        'model': model,
        'scaler': scaler,
        'class_names': class_names
    }

def evaluate_predictions(prediction_results, save_path=None):
    """
    Evaluate prediction results
    """
    print("\n" + "="*60)
    print("Evaluation of prediction results")
    print("="*60)
    predictions = prediction_results['predictions']
    probabilities = prediction_results['probabilities']
    # Calculate prediction confidence
    max_probabilities = np.max(probabilities, axis=1)
    avg_confidence = np.mean(max_probabilities)
    min_confidence = np.min(max_probabilities)
    max_confidence = np.max(max_probabilities)
    # Calculate prediction distribution
    unique_predictions, prediction_counts = np.unique(predictions, return_counts=True)
    prediction_distribution = dict(zip(unique_predictions, prediction_counts))
    # Print evaluation results
    print("\n Prediction quality evaluation:")
    print("-" * 30)
    print(f"Average confidence: {avg_confidence:.4f}")
    print(f"Minimum confidence: {min_confidence:.4f}")
    print(f"Maximum confidence: {max_confidence:.4f}")
    print("\n Prediction distribution:")
    print("-" * 30)
    for pred, count in prediction_distribution.items():
        percentage = (count / len(predictions)) * 100
        print(f"Class {pred}: {count} samples ({percentage:.1f}%)")
    # Save evaluation results
    if save_path:
        evaluation_data = {
            'prediction_metrics': {
                'avg_confidence': float(avg_confidence),
                'min_confidence': float(min_confidence),
                'max_confidence': float(max_confidence),
                'prediction_distribution': {str(k): int(v) for k, v in prediction_distribution.items()}
            }
        }
        json_path = save_path.replace('.png', '.json') if save_path.endswith('.png') else save_path + '.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
    return {
        'prediction_metrics': {
            'avg_confidence': avg_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence,
            'prediction_distribution': prediction_distribution
        }
    }

def visualize_predictions(prediction_results, save_path=None):
    """
    Visualize prediction results
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Pseudo-label semi-supervised learning prediction results visualization', fontsize=16, fontweight='bold')
    predictions = prediction_results['predictions']
    probabilities = prediction_results['probabilities']
    original_data = prediction_results['original_data']
    ax1 = axes[0, 0]
    unique_predictions, prediction_counts = np.unique(predictions, return_counts=True)
    bars = ax1.bar(unique_predictions, prediction_counts, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Predicted class')
    ax1.set_ylabel('Sample number')
    ax1.set_title('Prediction distribution')
    ax1.grid(True, alpha=0.3)
    for bar, count in zip(bars, prediction_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    ax2 = axes[0, 1]
    max_probabilities = np.max(probabilities, axis=1)
    ax2.hist(max_probabilities, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Prediction confidence')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Prediction confidence distribution (average: {np.mean(max_probabilities):.3f})')
    ax2.grid(True, alpha=0.3)
    ax3 = axes[1, 0]
    if original_data.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(original_data[:1000])  
    else:
        features_2d = original_data[:1000]
    scatter = ax3.scatter(features_2d[:, 0], features_2d[:, 1], 
                        c=predictions[:1000], cmap='tab10', alpha=0.6)
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.set_title('Feature space visualization')
    plt.colorbar(scatter, ax=ax3, label='Predicted class')
    ax4 = axes[1, 1]
    if len(unique_predictions) > 1:
        confidence_by_class = []
        class_labels = []
        for pred_class in unique_predictions:
            class_mask = predictions == pred_class
            class_confidences = max_probabilities[class_mask]
            confidence_by_class.append(class_confidences)
            class_labels.append(f'Class {pred_class}')
        ax4.boxplot(confidence_by_class, labels=class_labels)
        ax4.set_ylabel('Prediction confidence')
        ax4.set_title('Prediction confidence distribution by class')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

def save_predictions_to_txt(sample_names, prediction_results, output_path):
    """Save results to disk."""
    predictions = prediction_results['predictions']
    predicted_labels = prediction_results['predicted_labels']
    probabilities = prediction_results['probabilities']
    class_names = prediction_results['class_names']
    results_df = pd.DataFrame({
        'Sample_Name': sample_names,
        'Predicted_Class': predicted_labels
        # 'Predicted_Class_Index': predictions
    })
    # for i, class_name in enumerate(class_names):
    #     results_df[f'Probability_{class_name}'] = probabilities[:, i]
    results_df['Probability'] = np.max(probabilities, axis=1)  
    # results_df['Confidence_Level'] = results_df['Max_Probability'].apply(
    #     lambda x: 'High' if x > 0.8 else 'Medium' if x > 0.6 else 'Low'
    # )
    results_df.to_csv(output_path, sep='\t', index=False)
    class_counts = results_df['Predicted_Class'].value_counts()
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} samples")

if __name__ == "__main__":
    """
    predict pseudo model
    cmd:  D:/Anaconda3/envs/pt37/python.exe f:/breeding/code/my_code/multi-omics/predict_pseudo.py
    output: pseudo_prediction_results_*.npy, pseudo_prediction_results.png, pseudo_prediction_results.json
    """
    parser = argparse.ArgumentParser(description='pseudo label semi-supervised learning model prediction script')
    parser.add_argument('--model_path', type=str, default='pseudo_model.pth', help='model file path')
    parser.add_argument('--scaler_path', type=str, default='pseudo_scaler.pkl', help='scaler file path')
    parser.add_argument('--data_path', type=str, default='../../data/train_example_semi/train_example_semi_test.txt', help='new data file path (txt format)')
    parser.add_argument('--evaluate', action='store_true', default=1, help='evaluate prediction results')
    parser.add_argument('--visualize', action='store_true', default=0, help='visualize prediction results')
    parser.add_argument('--save_results', action='store_true', default=1, help='save prediction results')
    parser.add_argument('--output_path', type=str, default='pseudo_prediction_results', help='output file path prefix')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    set_random_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.data_path:
        import pandas as pd
        data_df = pd.read_csv(args.data_path, sep="\t")
        data_df = data_df.dropna()
        sample_names = data_df.iloc[:, 0].values  
        data = data_df.iloc[:, 1:-1].values.astype(float)  
        labels = data_df.iloc[:, -1].values  
    print(f"Data shape: {data.shape}")
    print(f"Sample number: {len(sample_names)}")
    print(f"Feature dimension: {data.shape[1]}")
    try:
        prediction_results = load_and_predict(
            args.model_path, args.scaler_path, data, device, args
        )
    except Exception as e:
        print(f"Prediction failed: {e}")
        exit(0)
    if args.evaluate:
        metrics = evaluate_predictions(
            prediction_results, 
            args.output_path if args.save_results else None
        )
    if args.visualize:
        visualize_predictions(
            prediction_results,
            args.output_path + '.png' if args.save_results else None
        )
    if args.save_results:
        save_predictions_to_txt(sample_names, prediction_results, args.output_path + '.txt')
