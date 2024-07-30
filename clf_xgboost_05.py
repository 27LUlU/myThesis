import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, confusion_matrix, recall_score, auc, roc_curve
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
import shap
import seaborn as sns
import matplotlib.ticker as mticker


def resampling(X, y):
    sm = ADASYN(random_state=0)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

directory = '/results'


df = pd.read_csv(f'{directory}/split_data/df_avg.csv')

df['label'] = df['label'].replace({0: 1, 1: 0})
X = df.drop(['label', 'ID', '57', '58', '59', '60', '61', '62', '55', '54', '56',
             '1', '2', '3', '4', '5'], axis=1)

y = df['label']

# scale data
scaler = StandardScaler()
X_sca = scaler.fit_transform(X)


param_distributions = {
    'n_estimators': randint(50, 3000),
    'max_depth': randint(1, 6),
    'eta': uniform(0.01, 0.3),
    'min_child_weight': randint(1, 10),
    'reg_alpha': uniform(0, 1),
}

# RandomizedSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
random_search = RandomizedSearchCV(XGBClassifier(scale_pos_weight=10, random_state=0),
                                   param_distributions,
                                   n_iter=50,
                                   cv=cv,
                                   scoring='f1_micro',
                                   random_state=0,
                                   n_jobs=-1,
                                   return_train_score=True)


x_, y_ = resampling(X_sca, y)
random_search.fit(x_, y_)
best_parameters = random_search.best_params_
# output the best parameters
print(f"Best parameters: {random_search.best_params_}")

# train the model
clf = XGBClassifier(
    scale_pos_weight=10,
    n_estimators=best_parameters['n_estimators'],
    max_depth=best_parameters['max_depth'],
    eta=best_parameters['eta'],
    min_child_weight=best_parameters['min_child_weight'],
    reg_alpha=best_parameters['reg_alpha'],
    random_state=0
)

# evaluate the model
loo = LeaveOneOut()
y_true = []
y_pred = []
y_pred_prob = []
shap_values = []
for train_index, test_index in loo.split(X_sca):
    X_train, X_test = X_sca[train_index], X_sca[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train, y_train = resampling(X_train, y_train)
    y_train = np.array(y_train)

    clf.fit(X_train, y_train)
    # predict
    y_pred.append(clf.predict(X_test)[0])
    y_true.append(y_test.values[0])
    y_pred_prob.append(clf.predict_proba(X_test)[:,1])

    # Create a SHAP explainer object
    explainer = shap.TreeExplainer(clf)

    # Calculate SHAP values for a sample of data (e.g., first 100 instances)
    shap_values.append(explainer.shap_values(X_test)[0])



# Plot the SHAP values
shap_values = np.array(shap_values)
X_sca2 = pd.DataFrame(X_sca)
feature_names=['sedentary aggregate duration time', 'light aggregate duration time', 'moderate-to-vigorous aggregate duration time', 'vigorous aggregate duration time',
               'sedentary max duration time', 'light max duration time', 'moderate-to-vigorous max duration time', 'vigorous max duration time',
               'sedentary sd duration time', 'light sd duration time', 'moderate-to-vigorous sd duration time', 'vigorous sd duration time',
               'sedentary mean duration time', 'light mean duration time', 'moderate-to-vigorous mean duration time', 'vigorous mean duration time',
               'sedentary ratio duration time', 'light ratio duration time', 'moderate-to-vigorous ratio duration time', 'vigorous ratio duration time',
               'sedentary aggregate count', 'light aggregate count', 'moderate-to-vigorous aggregate count', 'vigorous aggregate count',
               'sedentary max count', 'light max count', 'moderate-to-vigorous max count', 'vigorous max count',
               'sedentary sd count', 'light sd count', 'moderate-to-vigorous sd count', 'vigorous sd count',
               'sedentary mean count', 'light mean count', 'moderate-to-vigorous mean count', 'vigorous mean count',
               'sedentary ratio count', 'light ratio count', 'moderate-to-vigorous ratio count', 'vigorous ratio count',
               'minute-based mean count', 'minute-based sd count', 'minute-based cv count', 'minute-based max count', 'minute-based min count', 'minute-based p25 count', 'minute-based p50 count', 'minute-based p75 count',
               'spectral']

X.columns = list(range(shap_values.shape[1]))
# shap_bar = shap.summary_plot(shap_values, X_sca,
#                   plot_type="bar",
#                   plot_size=[15, 15],
#                   feature_names=feature_names,
#                   max_display=20,
#                   show=False)
# shap_dot = shap.summary_plot(shap_values, X_sca,
#                   plot_type="dot",
#                   plot_size=[15, 15],
#                   feature_names=feature_names,
#                   max_display=20,
#                   show=False)
# plt.savefig('plot/shap_bar.png')
# plt.savefig('plot/shap_dot.png')


# metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp)

report = classification_report(y_true, y_pred, output_dict=True)

print(f'acc:{accuracy:.4f}; precision:{precision:.4f}; recall:{recall:.4f}; specificity:{specificity:.4f}; f1:{f1:.4f}; cm: {cm}')


# plot confusion matrix
cm_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
row_sums = cm.sum(axis=1, keepdims=True)
cm_percentage = cm / row_sums * 100

# Create labels for the heatmap annotations with percentage values
cm_percentage_new = np.array([["{:.2f}%".format(value) for value in row] for row in cm_percentage])
labels = [f'{v1}\n{v2}' for v1, v2 in zip(cm_counts, cm_percentage_new.flatten())]
labels = np.asarray(labels).reshape(2,2)
# Create the heatmap
ax = sns.heatmap(cm_percentage, annot=labels, fmt='', cmap='Blues', cbar_kws={'format': '%.0f%%'})
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
ax.set(xlabel='Predicted label', ylabel='True label')
ax.set_xticklabels(['Problem walking', 'Normal walking'])
ax.set_yticklabels(['Problem walking', 'Normal walking'])
# plt.savefig('plot/cm.png')
# plt.show()

# plot roc
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)


# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='#F4A582', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='#4393C3', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
# plt.savefig('plot/roc.png')
# plt.show()


