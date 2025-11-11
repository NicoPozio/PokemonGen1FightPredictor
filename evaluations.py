import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, accuracy_score
)
from config import Config

def evaluate_model_performance(y_true, y_pred_cv_default, tuned_params):
    """
    Performs a detailed model evaluation from pre-computed CV predictions.
    Prints reports, saves confusion matrix.
    
    NOTE:
    - 'y_true' is the ground truth (y).
    - 'y_pred_cv_default' is the pre-computed 0/1 predictions.
    """
    print("\n--- Classification Report (Threshold 0.5) ---")
    print(classification_report(y_true, y_pred_cv_default, target_names=['Loss (0)', 'Win (1)']))

    print("\n--- Confusion Matrix (Threshold 0.5) ---")
    cm_tuned = confusion_matrix(y_true, y_pred_cv_default)
    disp_tuned = ConfusionMatrixDisplay(confusion_matrix=cm_tuned, display_labels=['Loss', 'Win'])
    disp_tuned.plot(cmap=plt.cm.Blues)
    model_name = tuned_params.get('model', 'Model')
    plt.title(f"Confusion Matrix (CV, Model={model_name})")
    
    output_filename = 'confusion_matrix_winner.png'
    plt.savefig(output_filename)
    print(f"Confusion Matrix saved as '{output_filename}'")
    plt.show()
    


def find_optimal_threshold(y_true, y_proba_cv_positive_tuned, y_pred_cv_default, tuned_params):
    """
    Analyzes the ROC Curve, calculates AUC, and finds the optimal threshold
    from pre-computed CV probabilities.
    
    NOTE:
    - 'y_true' is the ground truth (y).
    - 'y_proba_cv_positive_tuned' is the pre-computed probabilities for class 1.
    - 'y_pred_cv_default' is the pre-computed 0/1 predictions (at 0.5).
    """
    print("\n--- ROC Curve and Optimal Threshold Analysis ---")

    auc_score_tuned = roc_auc_score(y_true, y_proba_cv_positive_tuned)
    print(f"Area Under the Curve (AUC): {auc_score_tuned:.4f}")

    fpr_tuned, tpr_tuned, thresholds_tuned = roc_curve(y_true, y_proba_cv_positive_tuned)
    # Find the point closest to (0, 1) on the ROC curve
    distances_tuned = (fpr_tuned - 0)**2 + (tpr_tuned - 1)**2
    best_point_index_tuned = np.argmin(distances_tuned)
    optimal_threshold_tuned = thresholds_tuned[best_point_index_tuned]
    print(f"\n'Optimal' threshold found (closest to 0,1): {optimal_threshold_tuned:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_tuned, tpr_tuned, color='blue', label=f'ROC Curve (AUC = {auc_score_tuned:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Model (AUC = 0.5)')
    plt.scatter(fpr_tuned[best_point_index_tuned], tpr_tuned[best_point_index_tuned], color='green', zorder=5,
                label=f'Optimal Threshold (~{optimal_threshold_tuned:.2f})')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    model_name = tuned_params.get('model', 'Model')
    plt.title(f"ROC Curve (CV, Model={model_name})")
    plt.legend()
    plt.grid(True)
    
    output_filename = 'roc_curve_winner.png'
    plt.savefig(output_filename)
    print(f"ROC Curve saved as '{output_filename}'")
    plt.show()

    print("\n--- Threshold Performance Comparison ---")
    accuracy_default_tuned = accuracy_score(y_true, y_pred_cv_default)
    print(f"Accuracy @ 0.5 Threshold (Default): {accuracy_default_tuned:.4f}")
    
    y_pred_optimal_threshold_tuned = (y_proba_cv_positive_tuned >= optimal_threshold_tuned).astype(int)
    accuracy_optimal_tuned = accuracy_score(y_true, y_pred_optimal_threshold_tuned)
    print(f"Accuracy @ Optimal Threshold ({optimal_threshold_tuned:.4f}):   {accuracy_optimal_tuned:.4f}")

    print("\n--- Classification Report with OPTIMAL THRESHOLD ---")
    print(classification_report(y_true, y_pred_optimal_threshold_tuned, target_names=['Loss (0)', 'Win (1)']))
    
    return optimal_threshold_tuned


def create_submission_file(model_final, X_test, test_df, threshold, tuned_params, filename = 'submission.csv'):
    """
    Generates predictions on the test set and creates the submission file.
    
    NOTE:
    - 'model_final' is the FINAL trained Pipeline.
    - 'X_test' are the NON-SCALED test data (X_test_prescale).
    """
    print(f"\n Generating Submission")
    model_name = tuned_params.get('model', 'Model')
    print(f"Using Winning Model: {model_name}")
    print(f"Using Optimal Threshold: {threshold:.4f}")

    test_probabilities = model_final.predict_proba(X_test)[:, 1]
    final_predictions = (test_probabilities >= threshold).astype(int)

    submission = pd.DataFrame({
        Config.ID_COLUMN_NAME: test_df[Config.ID_COLUMN_NAME],
        Config.TARGET_COLUMN_NAME: final_predictions
    })

    submission.to_csv(filename, index=False)
    print(f"\n{filename} file created successfully!")
    
    print("Submission file head:")
    print(submission.head())
