# **Final Report: Insurance Claim Prediction**

##  **1. Project Objective**
The goal of this project is to predict whether an insurance customer will file a claim based on demographic, vehicle, and policy-related attributes.
This is a **binary classification problem** (Claim = 1, No Claim = 0).

---

##  **2. Dataset Overview**
* **Rows:** 595,212
* **Columns:** 59 (including target)
* **Target variable:** `target`
* **Claim distribution:**
  * 0 → 96.35% (No Claim)
  * 1 → 3.64% (Claim)
    → **Highly imbalanced dataset**
---

##  **3. Data Preprocessing Steps**
**Missing values:**
* Replaced all `-1` with `NaN`.
* Imputed missing values using **median** for numerical features.
**Outlier treatment:**
* Used **IQR (Interquartile Range)** capping to limit extreme outliers.
**Scaling:**
* Applied **StandardScaler** to normalize numeric columns.
**Encoding:**
* Used **One-Hot Encoding** for categorical columns → expanded dataset to **215 features**.
**Imbalance handling:**
* Applied **SMOTE** to balance classes (from 3.6% → 50%).
---

## **4. Model Training**
Built and evaluated several models — both **baseline** and **balanced** versions with class-weighting and SMOTE.

### Models Used:
* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* CatBoost
* Artificial Neural Network (ANN)
---

## **5. Model Tuning Techniques**
* **GridSearchCV / RandomizedSearchCV** for classical ML models
* **Learning rate, depth, and estimators** tuned for XGBoost, LightGBM, CatBoost
* **RandomSearch (Keras Tuner)** for ANN hyperparameter tuning
* **Class weighting and threshold optimization** for handling imbalance
---

## **6. Model Evaluation Metrics**
Used the following metrics:
* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* KS Statistic (Kolmogorov–Smirnov)
---

## **7. Final Model Leaderboard — Balanced Models**
| Model                            | Accuracy | Precision |  Recall  | F1-Score |   ROC-AUC  | KS Statistic | Remarks                                     |
| :------------------------------- | :------: | :-------: | :------: | :------: | :--------: | :----------: | :------------------------------------------ |
| **Balanced CatBoost**            |   0.65   |    0.05   | **0.54** |   0.10   | **0.6376** |   **0.193**  | Best balance of recall and AUC              |
| **Balanced XGBoost**             |   0.68   |    0.06   |   0.50   | **0.10** |   0.6349   |     0.192    | Stable and efficient                        |
| **Balanced Logistic Regression** |   0.63   |    0.05   | **0.56** |   0.09   |   0.6311   |     0.190    | Strong recall, interpretable                |
| **Balanced Random Forest**       | **0.69** |    0.06   |   0.47   |   0.10   |   0.6289   |     0.181    | High accuracy, slightly conservative        |
| **Balanced LightGBM**            | **0.75** |    0.06   |   0.40   |   0.10   |   0.6206   |     0.175    | High accuracy, weaker recall                |
| **Balanced ANN**                 |   0.60   |    0.05   |   0.55   |   0.09   |   0.6126   |     0.178    | Similar recall to logistic, slower training |

---

## **8. Key Insights**
1. **Balanced CatBoost** achieved the **highest KS (0.193)** and **ROC-AUC (0.6376)**, proving to be the most reliable model.
2. **Balanced Logistic Regression** achieved the **highest recall (0.56)** — good for risk detection tasks.
3. **Tree-based models (CatBoost, XGBoost, LightGBM)** outperformed ANN and logistic regression in overall discrimination.
4. **Balancing techniques** (SMOTE + class weights) significantly improved minority class detection.
5. Outlier capping and scaling helped stabilize ANN and gradient boosting models.

---

## **9. Feature Importance Summary**
### *Balanced CatBoost* — Top Predictors
* `ps_ind_03`, `ps_car_13`, `ps_ind_15`, `ps_reg_01`, `ps_reg_03`, `ps_car_15`, `ps_car_12`, `ps_car_14`
### *Balanced XGBoost*
* `ps_ind_06_bin`, `ps_ind_17_bin`, `ps_ind_05_cat_0`, `ps_car_07_cat_1`, `ps_reg_02`
### *Balanced LightGBM*
* `ps_car_13`, `ps_reg_03`, `ps_ind_03`, `ps_ind_15`, `ps_calc_10`, `ps_calc_14`
### *Balanced Logistic Regression*
* Key weights observed for: `ps_ind_04_cat_0`, `ps_car_02_cat_1`, `ps_car_11_cat_21`
---

## **10. Visual Analysis**
* ROC Curves → CatBoost > XGBoost > Logistic Regression
* KS Curves → Highest KS for CatBoost (~0.193)
* Precision-Recall Curves → Consistent improvement after balancing
* Bar Chart → ROC-AUC vs KS across all models
---

## **11. Conclusion**

* The **Balanced CatBoost model** is recommended for deployment due to:
  * Strong class separation (highest KS and ROC-AUC)
  * Better recall for detecting potential claimants
  * Efficient training and robust performance with categorical features

* **Model Saving:** All tuned and balanced models, along with preprocessing steps, were saved using `joblib` and `.keras` formats for future prediction.
---