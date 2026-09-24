from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, mean_absolute_error,
    mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

BASE_DIR = Path(__file__).resolve().parent
CHART_DIR = BASE_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)
CSV_PATH = BASE_DIR / "titanic.csv"
REPORT_PATH = BASE_DIR / "generated_report.md"
RANDOM_STATE = 42

# ---------- Helpers ----------
def save_chart(name):
    plt.tight_layout(); plt.savefig(CHART_DIR / name, dpi=150, bbox_inches="tight"); plt.close()

def section(title, body):
    return f"\n## {title}\n\n{body.strip()}\n"

def pct(x): return f"{100*x:.2f}%"

def evaluate_classifier(name, pipe, X_train, X_test, y_train, y_test):
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "AUC": roc_auc_score(y_test, prob),
        "ConfusionMatrix": confusion_matrix(y_test, pred),
        "FPR": roc_curve(y_test, prob)[0],
        "TPR": roc_curve(y_test, prob)[1],
        "Pipeline": pipe,
    }

# ---------- 1. Load exactly once and save offline fallback ----------
if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)
    load_note = "Loaded the committed offline fallback titanic.csv."
else:
    # The only sns.load_dataset call across this module.
    df = sns.load_dataset("titanic")
    df.to_csv(CSV_PATH, index=False)
    load_note = "Loaded Titanic once through sns.load_dataset and saved titanic.csv immediately."

report = ["# Module 2 Generated Results", f"\n{load_note}\n"]
print("Shape:", df.shape)
print(df.info())
print(df.describe(include="all"))

missing_pct = (df.isna().mean()*100).loc[lambda s:s>0].sort_values(ascending=False)
report.append(section("Dataset profile", f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**.\n\nMissing-value percentages:\n\n{missing_pct.rename('Missing %').to_frame().to_markdown()}"))

# ---------- 2. Threshold-based EDA cleaning ----------
eda_df = df.copy()
cleaning_notes=[]
for col, rate in missing_pct.items():
    if rate < 5:
        before=len(eda_df); eda_df=eda_df.dropna(subset=[col]).copy()
        cleaning_notes.append(f"- **{col}: {rate:.2f}% missing.** Under 5%, so rows missing this field were dropped ({before-len(eda_df)} rows at this stage).")
    elif rate <= 30:
        if pd.api.types.is_numeric_dtype(eda_df[col]):
            val=eda_df[col].median(); eda_df[col]=eda_df[col].fillna(val)
            cleaning_notes.append(f"- **{col}: {rate:.2f}% missing.** Between 5% and 30%, so median imputation was used ({val:.2f}).")
        else:
            val=eda_df[col].mode(dropna=True)[0]; eda_df[col]=eda_df[col].fillna(val)
            cleaning_notes.append(f"- **{col}: {rate:.2f}% missing.** Between 5% and 30%, so mode imputation was used ({val}).")
    else:
        eda_df[col]=eda_df[col].astype('object').fillna('Missing')
        cleaning_notes.append(f"- **{col}: {rate:.2f}% missing.** Missingness was too high for reliable imputation, so 'Missing' was encoded as a separate category.")
report.append(section("Missing-value handling", "\n".join(cleaning_notes)))

# ---------- 3. Univariate analysis ----------
outlier_notes=[]
for col in ["age","fare"]:
    q1,q3=eda_df[col].quantile([.25,.75]); iqr=q3-q1
    mask=(eda_df[col] < q1-1.5*iqr) | (eda_df[col] > q3+1.5*iqr)
    outlier_notes.append(f"- **{col}: {int(mask.sum())} IQR outliers** using [{q1-1.5*iqr:.2f}, {q3+1.5*iqr:.2f}].")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    sns.histplot(data=eda_df,x=col,kde=True,ax=axes[0]); axes[0].set_title(f"{col.title()} histogram")
    sns.boxplot(data=eda_df,x=col,ax=axes[1]); axes[1].set_title(f"{col.title()} box plot")
    save_chart(f"{col}_hist_box.png")
fare_mean=eda_df.fare.mean(); fare_median=eda_df.fare.median(); fare_mode=eda_df.fare.mode().iloc[0]
if fare_mean > fare_median: skew="right-skewed"
elif fare_mean < fare_median: skew="left-skewed"
else: skew="approximately symmetric"
outlier_notes.append(f"- Fare mean = **{fare_mean:.2f}**, median = **{fare_median:.2f}**, mode = **{fare_mode:.2f}**. Because the mean is {'greater than' if fare_mean>fare_median else 'less than or equal to'} the median, fare is **{skew}**.")
report.append(section("Univariate analysis", "\n".join(outlier_notes)))

# ---------- 4. Bivariate analysis with boolean masking ----------
def rate(mask): return eda_df.loc[mask,"survived"].mean()
sex_rates={s:rate(eda_df.sex==s) for s in sorted(eda_df.sex.unique())}
pclass_rates={int(c):rate(eda_df.pclass==c) for c in sorted(eda_df.pclass.unique())}
combo=[]
for s in sorted(eda_df.sex.unique()):
    for c in sorted(eda_df.pclass.unique()):
        combo.append({"sex":s,"pclass":int(c),"survival_rate":rate((eda_df.sex==s)&(eda_df.pclass==c))})
sex_table=pd.Series(sex_rates,name="Survival rate").to_frame()
pclass_table=pd.Series(pclass_rates,name="Survival rate").to_frame()
combo_df=pd.DataFrame(combo)

corr_cols=["survived","pclass","age","sibsp","parch","fare"]
corr=eda_df[corr_cols].corr()
plt.figure(figsize=(8,6)); sns.heatmap(corr,annot=True,cmap="coolwarm",center=0,fmt=".2f"); plt.title("Correlation matrix: required six columns")
save_chart("correlation_heatmap.png")
pairs=[]
for i,a in enumerate(corr.columns):
    for j,b in enumerate(corr.columns):
        if i<j: pairs.append((a,b,corr.loc[a,b],abs(corr.loc[a,b])))
pairs=sorted(pairs,key=lambda x:x[3],reverse=True)[:2]
report.append(section("Bivariate analysis", f"### Survival rate by sex\n\n{sex_table.to_markdown()}\n\n### Survival rate by passenger class\n\n{pclass_table.to_markdown()}\n\n### Survival rate by sex and passenger class\n\n{combo_df.to_markdown(index=False)}\n\nThe two strongest absolute off-diagonal correlations are **{pairs[0][0]} vs {pairs[0][1]} ({pairs[0][2]:.3f})** and **{pairs[1][0]} vs {pairs[1][1]} ({pairs[1][2]:.3f})**."))

# ---------- 5. Four multivariate charts ----------
plt.figure(figsize=(7,4)); sns.barplot(data=eda_df,x="sex",y="survived",errorbar=None); plt.title("Survival rate by sex"); save_chart("story_1_survival_sex.png")
plt.figure(figsize=(7,4)); sns.barplot(data=eda_df,x="pclass",y="survived",hue="sex",errorbar=None); plt.title("Survival rate by class and sex"); save_chart("story_2_class_sex.png")
plt.figure(figsize=(8,4)); sns.boxplot(data=eda_df,x="pclass",y="fare",hue="survived"); plt.title("Fare by class and survival"); save_chart("story_3_fare_class_survival.png")
plt.figure(figsize=(8,4)); sns.histplot(data=eda_df,x="age",hue="survived",element="step",stat="density",common_norm=False); plt.title("Age distribution by survival"); save_chart("story_4_age_survival.png")
story=f"""1. **Survival by sex:** The chart compares average survival rates across sex categories. The numeric values in the table above provide the exact comparison; the chart makes the gap easier to see.
2. **Class and sex together:** Survival varies across both passenger class and sex, showing that one feature alone does not fully describe the pattern. The combined breakdown is a better multivariate explanation than class-only or sex-only views.
3. **Fare, class and survival:** Fare distributions differ strongly across passenger classes, so fare partly carries class information. The survival split within each class helps assess whether fare still separates outcomes after class is considered.
4. **Age and survival:** The overlapping distributions show that age alone does not perfectly separate survivors and non-survivors. Age can still contribute useful information when combined with sex, class, family size and fare."""
report.append(section("Multivariate data story", story))

# ---------- 6. Exploratory z-score check ----------
std_check=eda_df[["age","fare"]].copy()
for col in ["age","fare"]: std_check[col+"_z"]=(std_check[col]-std_check[col].mean())/std_check[col].std(ddof=0)
summary=pd.DataFrame({"before_mean":eda_df[["age","fare"]].mean(),"before_std":eda_df[["age","fare"]].std(ddof=0),"after_mean":std_check[["age_z","fare_z"]].mean().values,"after_std":std_check[["age_z","fare_z"]].std(ddof=0).values},index=["age","fare"])
report.append(section("EDA standardization check", summary.to_markdown()))

# ---------- 7-10. Split, preprocessing, three classifiers ----------
features=["pclass","sex","age","sibsp","parch","fare","embarked"]
X=df[features].copy(); y=df["survived"].astype(int)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=RANDOM_STATE)
class_balance=y.value_counts(normalize=True).sort_index()

numeric=["pclass","age","sibsp","parch","fare"]
categorical=["sex","embarked"]
num_pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
cat_pipe=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
preprocessor=ColumnTransformer([("num",num_pipe,numeric),("cat",cat_pipe,categorical)],sparse_threshold=0)
models={
"Logistic Regression":LogisticRegression(max_iter=1000,random_state=RANDOM_STATE),
"Decision Tree":DecisionTreeClassifier(max_depth=4,random_state=RANDOM_STATE),
"Random Forest":RandomForestClassifier(n_estimators=200,random_state=RANDOM_STATE,oob_score=True)
}
results=[]
for name,model in models.items():
    pipe=Pipeline([("preprocessor",preprocessor),("model",model)])
    results.append(evaluate_classifier(name,pipe,X_train,X_test,y_train,y_test))
metrics_df=pd.DataFrame([{k:r[k] for k in ["Model","Accuracy","Precision","Recall","F1","AUC"]} for r in results]).set_index("Model")
metrics_df.to_csv(BASE_DIR/"classification_metrics.csv")

fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,r in zip(axes,results): sns.heatmap(r["ConfusionMatrix"],annot=True,fmt="d",cmap="Blues",cbar=False,ax=ax); ax.set_title(r["Model"]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
save_chart("confusion_matrices.png")
plt.figure(figsize=(7,5))
for r in results: plt.plot(r["FPR"],r["TPR"],label=f'{r["Model"]} AUC={r["AUC"]:.3f}')
plt.plot([0,1],[0,1],'k--'); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves"); plt.legend(); save_chart("roc_curves.png")

dt_result=next(r for r in results if r["Model"]=="Decision Tree")
dt_pipe=dt_result["Pipeline"]
feature_names=dt_pipe.named_steps["preprocessor"].get_feature_names_out()
plt.figure(figsize=(24,12)); plot_tree(dt_pipe.named_steps["model"],feature_names=feature_names,class_names=["Not survived","Survived"],filled=True,rounded=True,max_depth=3,fontsize=8); plt.title("Decision Tree")
save_chart("decision_tree.png")
report.append(section("Classification", f"Class balance before splitting:\n\n{class_balance.rename('Proportion').to_frame().to_markdown()}\n\nA stratified split was used so train and test partitions preserve the target proportions. Preprocessing was fitted only through the training pipeline and test data was used only for transformation and evaluation.\n\n{metrics_df.to_markdown(floatfmt='.4f')}"))

# ---------- 11. Imbalance comparison ----------
imb=[]
base_pipe=Pipeline([("preprocessor",preprocessor),("model",LogisticRegression(max_iter=1000,random_state=RANDOM_STATE))])
bal_pipe=Pipeline([("preprocessor",preprocessor),("model",LogisticRegression(max_iter=1000,class_weight="balanced",random_state=RANDOM_STATE))])
smote_pipe=ImbPipeline([("preprocessor",preprocessor),("smote",SMOTE(random_state=RANDOM_STATE)),("model",LogisticRegression(max_iter=1000,random_state=RANDOM_STATE))])
for name,pipe in [("Baseline",base_pipe),("Class weight balanced",bal_pipe),("SMOTE training only",smote_pipe)]:
    pipe.fit(X_train,y_train); pred=pipe.predict(X_test)
    imb.append({"Strategy":name,"Precision":precision_score(y_test,pred),"Recall":recall_score(y_test,pred),"F1":f1_score(y_test,pred)})
imb_df=pd.DataFrame(imb).set_index("Strategy"); imb_df.to_csv(BASE_DIR/"imbalance_comparison.csv")
best_imb=imb_df["F1"].idxmax()
report.append(section("Imbalance handling comparison", f"{imb_df.to_markdown(floatfmt='.4f')}\n\n**Conclusion:** On this split, **{best_imb}** produced the highest F1 score. The final choice should consider both recall and precision because a higher recall may come with more false positives."))

# ---------- 12. Random Forest tuning and OOB ----------
rf_tune=Pipeline([("preprocessor",preprocessor),("model",RandomForestClassifier(oob_score=True,random_state=RANDOM_STATE))])
param_grid={"model__n_estimators":[100,200],"model__max_depth":[None,5,10],"model__max_features":["sqrt","log2"]}
grid=GridSearchCV(rf_tune,param_grid=param_grid,cv=5,scoring="f1",n_jobs=-1)
grid.fit(X_train,y_train)
best_rf=grid.best_estimator_; oob=best_rf.named_steps["model"].oob_score_
report.append(section("Random Forest tuning", f"Best parameters: `{grid.best_params_}`  \nBest cross-validation F1: **{grid.best_score_:.4f}**  \nOOB score: **{oob:.4f}**"))

# ---------- 13. Regression side-task ----------
reg_features=["pclass","sex","age","sibsp","parch","survived","embarked"]
Xr=df[reg_features].copy(); yr=df["fare"].astype(float)
Xr_train,Xr_test,yr_train,yr_test=train_test_split(Xr,yr,test_size=.2,random_state=RANDOM_STATE)
reg_num=["pclass","age","sibsp","parch","survived"]; reg_cat=["sex","embarked"]
reg_pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),reg_num),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),reg_cat)],sparse_threshold=0)
reg_pipe=Pipeline([("preprocessor",reg_pre),("model",LinearRegression())]); reg_pipe.fit(Xr_train,yr_train); reg_pred=reg_pipe.predict(Xr_test)
mae=mean_absolute_error(yr_test,reg_pred); rmse=np.sqrt(mean_squared_error(yr_test,reg_pred)); r2=r2_score(yr_test,reg_pred)
p=reg_pipe.named_steps["preprocessor"].transform(Xr_test).shape[1]; n=len(yr_test); adj_r2=1-(1-r2)*(n-1)/(n-p-1)
residuals=yr_test-reg_pred
plt.figure(figsize=(8,5)); sns.scatterplot(x=reg_pred,y=residuals); plt.axhline(0,color='red',linestyle='--'); plt.xlabel("Predicted fare"); plt.ylabel("Residual"); plt.title("Regression residual plot"); save_chart("regression_residuals.png")
# Objective aid only: compare residual variance across predicted-value quartiles.
bins=pd.qcut(pd.Series(reg_pred),q=4,duplicates='drop'); variances=pd.Series(residuals.to_numpy()).groupby(bins,observed=False).var(); ratio=variances.max()/variances.min() if variances.min()>0 else np.inf
hetero="shows evidence of non-constant variance" if ratio>=2 else "does not show strong evidence of non-constant variance"
reg_metrics=pd.DataFrame({"Model":["Linear Regression"],"MAE":[mae],"RMSE":[rmse],"R2":[r2],"Adjusted R2":[adj_r2]}).set_index("Model"); reg_metrics.to_csv(BASE_DIR/"regression_metrics.csv")
report.append(section("Regression side-task", f"{reg_metrics.to_markdown(floatfmt='.4f')}\n\nThe residual-variance quartile ratio was **{ratio:.2f}**. Using 2.0 as a transparent diagnostic threshold, the residual plot **{hetero}**. This is a visual/diagnostic conclusion rather than a formal statistical test."))

# ---------- 14-15. Recommendation, save/reload best full pipeline ----------
best_name=metrics_df["F1"].idxmax()
best_result=next(r for r in results if r["Model"]==best_name)
# Include tuned RF as a candidate too.
tuned_eval=evaluate_classifier("Tuned Random Forest",best_rf,X_train,X_test,y_train,y_test)
if tuned_eval["F1"] > best_result["F1"]: best_name="Tuned Random Forest"; best_pipeline=best_rf; best_metrics=tuned_eval
else: best_pipeline=best_result["Pipeline"]; best_metrics=best_result
joblib.dump(best_pipeline,BASE_DIR/"best_classification_pipeline.joblib")
loaded=joblib.load(BASE_DIR/"best_classification_pipeline.joblib")
reload_prediction=loaded.predict(X_test.head(3))
recommendation=f"""The recommended classifier is **{best_name}**. On the held-out test set it achieved accuracy **{best_metrics['Accuracy']:.4f}**, precision **{best_metrics['Precision']:.4f}**, recall **{best_metrics['Recall']:.4f}**, F1 **{best_metrics['F1']:.4f}**, and AUC **{best_metrics['AUC']:.4f}**. F1 was used as the primary selection measure because it balances precision and recall. The final decision should still consider the operational cost of false positives versus false negatives. The complete preprocessing-plus-model pipeline was saved and successfully reloaded; predictions for three raw rows were `{reload_prediction.tolist()}`."""
report.append(section("Final comparison and recommendation", "### Classification metrics\n\n"+metrics_df.to_markdown(floatfmt='.4f')+"\n\n### Regression metrics, separate scale\n\n"+reg_metrics.to_markdown(floatfmt='.4f')+"\n\n"+recommendation))

REPORT_PATH.write_text("\n".join(report),encoding="utf-8")
print("\nMODULE 2 COMPLETED")
print("Generated report:",REPORT_PATH)
print("Best pipeline:",BASE_DIR/"best_classification_pipeline.joblib")
