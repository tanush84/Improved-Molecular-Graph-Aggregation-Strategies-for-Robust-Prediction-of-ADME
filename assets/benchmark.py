import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from assets.functions import *
def run_descriptor_benchmark(
    train_df,
    val_df,
    test_df,
    target_col,
    descriptors=("FCFP6", "MACCS", "RDKIT", "MORDRED"),
    random_state=42
    ):
    '''
    # --------------------------------------------------
    # Metric function (version-safe)
    # --------------------------------------------------
    def evaluate(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred)
        }
    '''
    # --------------------------------------------------
    # Feature builder
    # --------------------------------------------------
    def featurize_df(df, method):
        X = []
        y = df[target_col].values.astype(np.float32)

        for smi in df["smiles"]:
            mol = smiles_to_mol(smi)
            if mol is None:
                X.append(None)
                continue

            if method == "FCFP6":
                X.append(calc_fcfp6(mol))
            elif method == "MACCS":
                X.append(calc_maccs(mol))
            elif method == "RDKIT":
                X.append(calc_rdkit_desc(mol))
            elif method == "MORDRED":
                X.append(calc_mordred(mol))
            else:
                raise ValueError(method)

        X = np.array(X, dtype=object)
        valid = np.array([x is not None for x in X])
        X = np.vstack(X[valid])
        y = y[valid]

        return X, y
    # --------------------------------------------------
    # Models
    # --------------------------------------------------
    models = {
        "RF": RandomForestRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1
        ),
        "SVM": SVR(
            kernel="rbf",
            C=10.0,
            gamma="scale"
        ),
        "XGB": XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1
        )
    }

    # --------------------------------------------------
    # Results storage
    # --------------------------------------------------
    results = []

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------
    for desc in descriptors:
        print(f"\n🔹 Featurizing with {desc}...")

        X_train, y_train = featurize_df(train_df, desc)
        X_val,   y_val   = featurize_df(val_df, desc)
        X_test,  y_test  = featurize_df(test_df, desc)

        for model_name, model in models.items():
            print(f"   ▶ Training {model_name}")

            # Conditional scaling
            if model_name in ["SVM", "XGB"]:
                pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", model)
                ])
            else:
                pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", model)
                ])

            # Train
            pipe.fit(X_train, y_train)

            # Validation
            val_pred = pipe.predict(X_val)
            val_metrics = evaluate(y_val, val_pred)

            # Test
            test_pred = pipe.predict(X_test)
            test_metrics = evaluate(y_test, test_pred)

            # Store
            results.append({
                "Descriptor": desc,
                "Model": model_name,
                "Val_MAE": val_metrics["MAE"],
                "Val_RMSE": val_metrics["RMSE"],
                "Val_R2": val_metrics["R2"],
                "Test_MAE": test_metrics["MAE"],
                "Test_RMSE": test_metrics["RMSE"],
                "Test_R2": test_metrics["R2"],
            })

    results_df = pd.DataFrame(results)
    return results_df
