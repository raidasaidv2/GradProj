import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd  # Import pandas for DataFrame operations
import dataframe_image as dfi  # Import dataframe-image for exporting DataFrame as an image

# === Create output folder ===
output_dir = "output_graphs"
os.makedirs(output_dir, exist_ok=True)

# === Load files ===
with open(r"C:\Users\raida\Downloads\gait_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(r"C:\Users\raida\Downloads\label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open(r"C:\Users\raida\Downloads\gait_dataset.pkl", "rb") as f:
    data = pickle.load(f)

# Update how data is unpacked
X, y = data  # Unpack the tuple into X (features) and y (labels)

y_pred = model.predict(X)

# Ensure class_names contains strings
class_names = [str(cls) for cls in label_encoder.classes_]

# === Confusion Matrix as a Table ===
cm = confusion_matrix(y, y_pred)
with open(os.path.join(output_dir, "confusion_matrix.txt"), "w") as f:
    f.write("Confusion Matrix:\n")
    f.write("\t" + "\t".join(class_names) + "\n")  # Header row
    for i, row in enumerate(cm):
        f.write(f"{class_names[i]}\t" + "\t".join(map(str, row)) + "\n")  # Rows

print("\nConfusion Matrix saved as a text file.")

# === Simplified Classification Report ===
report = classification_report(y, y_pred, target_names=class_names, output_dict=True)
simplified_report = "Class\tPrecision\tRecall\tF1-Score\n"
for cls, metrics in report.items():
    if cls in class_names:  # Include only class-specific metrics
        simplified_report += f"{cls}\t{metrics['precision']:.2f}\t{metrics['recall']:.2f}\t{metrics['f1-score']:.2f}\n"

# Save simplified report to a text file
with open(os.path.join(output_dir, "simplified_classification_report.txt"), "w") as f:
    f.write("Simplified Classification Report:\n")
    f.write(simplified_report)

print("\nSimplified Classification Report saved as a text file.")

# === Confusion Matrix for Top 9 Classes ===
# Pick top 9 most common labels
label_counts = Counter(y)
top_labels = [label for label, _ in label_counts.most_common(9)]

# Filter only those samples
mask = np.isin(y, top_labels)
filtered_y = y[mask]
filtered_pred = y_pred[mask]

cm = confusion_matrix(filtered_y, filtered_pred, labels=top_labels)
filtered_class_names = label_encoder.inverse_transform(top_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=filtered_class_names, yticklabels=filtered_class_names, cmap="YlGnBu")
plt.title("Confusion Matrix (Top Classes)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_top_classes.png"))
plt.close()

# === Short Classification Report ===
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='macro')
recall = recall_score(y, y_pred, average='macro')
f1 = f1_score(y, y_pred, average='macro')

short_report = f"""
Accuracy     : {accuracy:.2f}
Precision    : {precision:.2f}
Recall       : {recall:.2f}
F1 Score     : {f1:.2f}
"""

# Print and save
print(short_report)
plt.figure(figsize=(6, 3))
plt.axis('off')
plt.text(0, 1, short_report, fontsize=12, verticalalignment='top', family='monospace')
plt.title("Short Classification Report", loc='left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "short_classification_report.png"))
plt.close()

# === ROC Curve (macro-averaged) ===
y_bin = label_binarize(y, classes=np.arange(len(class_names)))
y_score = model.predict_proba(X)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Macro-average
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(class_names)):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= len(class_names)
macro_auc = auc(all_fpr, mean_tpr)

plt.figure(figsize=(8, 6))
plt.plot(all_fpr, mean_tpr, color='navy', label=f'Macro-average ROC (AUC = {macro_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-Averaged ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve_macro.png"))
plt.close()

# === Feature Importance (RandomForest) ===
if isinstance(model, RandomForestClassifier):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    plt.close()
else:
    print("⚠️ Feature importance not available for this model type.")

# === Classification Results as a Table ===
# Extract key metrics for each class
report = classification_report(y, y_pred, target_names=class_names, output_dict=True)
table_data = [["Class", "Precision", "Recall", "F1-Score"]]
for cls, metrics in report.items():
    if cls in class_names:  # Include only class-specific metrics
        table_data.append([cls, f"{metrics['precision']:.2f}", f"{metrics['recall']:.2f}", f"{metrics['f1-score']:.2f}"])

# Add overall metrics to the table
table_data.append(["Overall (Macro Avg)", f"{report['macro avg']['precision']:.2f}", f"{report['macro avg']['recall']:.2f}", f"{report['macro avg']['f1-score']:.2f}"])
table_data.append(["Overall (Weighted Avg)", f"{report['weighted avg']['precision']:.2f}", f"{report['weighted avg']['recall']:.2f}", f"{report['weighted avg']['f1-score']:.2f}"])

# Create a matplotlib table
plt.figure(figsize=(8, len(table_data) * 0.6))
plt.axis('off')
table = plt.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(table_data[0]))))
plt.title("Classification Results", fontsize=14, loc='center')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "classification_results_table.png"))
plt.close()

print("\nClassification Results Table saved as an image.")

# === Classification Summary ===
# Convert classification report to DataFrame
report_df = pd.DataFrame(report).transpose()

# Show only macro avg, weighted avg, accuracy
summary = report_df.loc[["accuracy", "macro avg", "weighted avg"]]

print(summary)

# Save as image
dfi.export(summary.round(2), os.path.join(output_dir, "classification_summary.png"))

print("\nClassification Summary saved as an image.")
