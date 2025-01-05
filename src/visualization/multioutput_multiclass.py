import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score
import seaborn as sns
from sklearn.metrics import classification_report
import numpy as np

def plot_distributions(y_true, y_pred):
    """
    Plot distribution histograms comparing true vs predicted values for each target.
    
    Args:
        y_true (DataFrame): True target values
        y_pred (DataFrame): Predicted target values
    """
    num_outputs = y_true.shape[1]
    nrows = (num_outputs + 1) // 2  # Calculate number of rows needed for 2 columns
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(15, 5 * nrows))
    axes = axes.ravel()

    for i, col in enumerate(y_true.columns):
        axes[i].hist(y_true.iloc[:, i], bins=5, alpha=0.5, label='Réel', color='blue', edgecolor='black')
        axes[i].hist(y_pred.iloc[:, i], bins=5, alpha=0.5, label='Prédit', color='orange', edgecolor='black')
        axes[i].set_title(f"Distribution Réelle vs Prédite - {col}")
        axes[i].legend()
    
    # Hide empty subplots if odd number of outputs
    if num_outputs % 2 != 0:
        axes[-1].set_visible(False)
        
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(y_true, y_pred):
    """
    Plot confusion matrices for each target variable.
    
    Args:
        y_true (DataFrame): True target values
        y_pred (DataFrame): Predicted target values
    """
    num_outputs = y_true.shape[1]
    nrows = (num_outputs + 1) // 2  # Calculate number of rows needed for 2 columns
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 5 * nrows))
    axes = axes.ravel()

    for i, col in enumerate(y_true.columns):
        cm = confusion_matrix(y_true.iloc[:, i], y_pred.iloc[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f"Matrice de Confusion - {col}")
        axes[i].set_xlabel("Prédit")
        axes[i].set_ylabel("Réel")

    # Hide empty subplots if odd number of outputs
    if num_outputs % 2 != 0:
        axes[-1].set_visible(False)

    plt.tight_layout()
    plt.show()

def classif_report(y_true, y_pred):
    """
    Print classification reports for each target variable.
    
    Args:
        y_true (DataFrame): True target values
        y_pred (DataFrame): Predicted target values
    """
    for i, col in enumerate(y_true.columns):
        print(f"Classification Report - {col}")
        print(classification_report(y_true.iloc[:, i], y_pred.iloc[:, i]))
    return

def plot_cross_validation_metrics(cv_results):
    """
    Plot cross-validation metrics including accuracy, hamming loss and jaccard scores.
    
    Args:
        cv_results (list): List of tuples (y_true, y_pred) containing true and predicted values for each fold
        
    Returns:
        None. Displays plots of the metrics.
    """
    # Calculate metrics for each fold
    accuracies = []
    hamming_losses = []
    jaccard_scores = []
    
    for y_true, y_pred in cv_results:
        # Calculate accuracy per target using sklearn's accuracy_score
        fold_accuracies = []
        fold_jaccard = []
        for i in range(y_true.shape[1]):
            fold_accuracies.append(accuracy_score(y_true.iloc[:,i], y_pred.iloc[:,i]))
            fold_jaccard.append(jaccard_score(y_true.iloc[:,i], y_pred.iloc[:,i], average="micro"))
        accuracies.append(fold_accuracies)
        
        # Calculate hamming loss using sklearn's hamming_loss
        # Calculate hamming loss per target and take average since multiclass-multioutput is not supported
        fold_hamming = []
        for i in range(y_true.shape[1]):
            fold_hamming.append(hamming_loss(y_true.iloc[:,i], y_pred.iloc[:,i]))
        hamming_losses.append(fold_hamming)
        
        # Calculate Jaccard scores
        jaccard_scores.append(fold_jaccard)
        
    accuracies = np.array(accuracies)
    jaccard_scores = np.array(jaccard_scores)
    hamming_losses = np.array(hamming_losses)

    # Plot metrics
    plt.figure(figsize=(15,5))

    # Plot accuracies for each target
    plt.subplot(1,3,1)
    for i, col in enumerate(cv_results[0][0].columns):
        plt.plot(accuracies[:,i], marker='o', label=col)
    plt.title('Accuracy per Target Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Hamming Loss per target
    plt.subplot(1,3,2)
    for i, col in enumerate(cv_results[0][0].columns):
        plt.plot(hamming_losses[:,i], marker='o', label=col)
    plt.title('Hamming Loss per Target Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Hamming Loss')
    plt.legend()

    # Plot Jaccard Scores for each target
    plt.subplot(1,3,3)
    for i, col in enumerate(cv_results[0][0].columns):
        plt.plot(jaccard_scores[:,i], marker='o', label=col)
    plt.title('Jaccard Score per Target Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Jaccard Score')
    plt.legend()

    plt.tight_layout()
    plt.show()