import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,matthews_corrcoef
import optuna
from ood_detection.classifier.feature_extractor import load_feature_extractor, build_features


def train_classifier(df: pd.DataFrame, model: str, feature_extractor: str,
                    return_metric: bool = False, df_val_ckpt: pd.DataFrame = None, 
                    n_splits: int = 5, skip_cv: bool = False,
                    **kwargs):
    if "text" not in df.columns:
        print("'text' should exist in the dataframe columns")
        return
    if "intent" not in df.columns:
        print("'intent' should exist in the dataframe columns")
        return
    
    if df_val_ckpt is not None:
        if "text" not in df_val_ckpt.columns:
            print("'text' should exist in the validation dataframe columns")
            return
        if "intent" not in df_val_ckpt.columns:
            print("'intent' should exist in the validation dataframe columns")
            return
        
    if "_best_ckpt" in feature_extractor and df_val_ckpt is None:
        print("df_val_ckpt is None but using '_best_ckpt' in feature_extractor. Make sure to pass validation data in the df_val_ckpt arguments")
        return 
    
    if "_best_ckpt" in feature_extractor and "_best_ckpt" not in model:
        print(f"_best_ckpt is found in 'feature_extractor' but not found in 'model'. Make sure to use the correct model name: {model}_best_ckpt")
        return
    elif "_best_ckpt" not in feature_extractor and "_best_ckpt" in model:
        print(f"_best_ckpt is found in 'model' but not found in 'feature_extractor'. Make sure to use the correct feature_extractor name: {feature_extractor}_best_ckpt")
        return

    cv_scores_dict = {"precision":[],"recall":[],"f1-score":[],"mcc":[],"support":[]}
    
    feature_model = load_feature_extractor(feature_extractor)
    
    if not skip_cv:
        #Perform Stratified K-Fold
        skf = StratifiedKFold(n_splits=n_splits)
        for train_idx, val_idx in tqdm(skf.split(df,df["intent"])):
            df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]

            if model not in ['biencoder']:
                # Generate Features
                output = build_features(feature_extractor, df_train["text"], df_train["intent"],
                                    df_val["text"], 
                                    df_val_ckpt["text"] if df_val_ckpt is not None else None,
                                    df_val_ckpt["intent"] if df_val_ckpt is not None else None,
                                    model=feature_model,
                                    **kwargs)

                if ("_best_ckpt" in feature_extractor) and (df_val_ckpt is not None):
                    output, output_ckpt = output
                    X_val_ckpt, y_val_ckpt = output_ckpt

                X_train, y_train, X_val = output    

            # Train Model
            if model == "gaussian_nb":
                from ood_detection.classifier.train_utils import fit_nb
                clf = fit_nb(X_train,y_train,nb_type = "gaussian")
            elif model == "multinomial_nb":
                from ood_detection.classifier.train_utils import fit_nb
                clf = fit_nb(X_train,y_train,nb_type = "multinomial")
            elif model == "svc":
                from ood_detection.classifier.train_utils import fit_svc
                clf = fit_svc(X_train,y_train)
            elif model == "rf":
                from ood_detection.classifier.train_utils import fit_rf
                clf = fit_rf(X_train,y_train)
            elif model == "mlp": 
                from ood_detection.classifier.train_utils import fit_mlp         
                clf = fit_mlp(feature_extractor,X_train,y_train,**kwargs)
            elif model == "mlp_best_ckpt":   
                from ood_detection.classifier.train_utils import fit_mlp         
                clf = fit_mlp(feature_extractor,X_train,y_train,X_val_ckpt, y_val_ckpt,**kwargs)
            elif model == "adb": 
                from ood_detection.classifier.train_utils import fit_adb       
                clf = fit_adb(feature_extractor,X_train,y_train)
            elif model == "mlp_dense_flipout": 
                from ood_detection.classifier.train_utils import fit_mlp_dense_flipout       
                clf = fit_mlp_dense_flipout(feature_extractor,X_train,y_train)
            elif model == "biencoder": 
                from ood_detection.classifier.train_utils import fit_biencoder       
                clf = fit_biencoder(feature_extractor,df_train)
            else:
                print("Model's not supported.")
                return

            if model not in ['biencoder']:
                val_pred = clf.predict(X_val)
            else:
                val_pred = clf.predict(df_val)
            report = classification_report(df_val["intent"],val_pred,
                                           zero_division=0,output_dict=True
                                          )

            report = report['weighted avg']
            for metric in report:
                cv_scores_dict[metric].append(report[metric])
            cv_scores_dict["mcc"].append(matthews_corrcoef(df_val['intent'],val_pred))

        assert np.sum(cv_scores_dict["support"]) == len(df)

        # Print Stratified K-Fold Mean Metrics        
        output_metric_dict = {}
        for metric,values in cv_scores_dict.items():
            mean_score = np.mean(values)
            std_score = np.std(values)
            print(f"{metric}: {round(mean_score,3)} +/- {round(std_score,3)}")
            output_metric_dict[f"mean_weighted_avg_{metric}"] = mean_score
            output_metric_dict[f"std_weighted_avg_{metric}"] = std_score
        
    if model not in ['biencoder']:
        # Generate Features from Full Data
        output = build_features(feature_extractor, df["text"], df["intent"],
                            None, 
                            df_val_ckpt["text"] if df_val_ckpt is not None else None,
                            df_val_ckpt["intent"] if df_val_ckpt is not None else None,
                            model=feature_model,
                            **kwargs)
        
        if ("_best_ckpt" in feature_extractor) and (df_val_ckpt is not None):
            output, output_ckpt = output
            X_val_ckpt, y_val_ckpt = output_ckpt

        X_full,y_full = output
    
    #Train on full data
    if model == "gaussian_nb":
        from ood_detection.classifier.train_utils import fit_nb
        clf_full = fit_nb(X_full,y_full,nb_type = "gaussian")
    elif model == "multinomial_nb":
        from ood_detection.classifier.train_utils import fit_nb
        clf_full = fit_nb(X_full,y_full,nb_type = "multinomial")
    elif model == "svc":
        from ood_detection.classifier.train_utils import fit_svc
        clf_full = fit_svc(X_full,y_full)
    elif model == "rf":
        from ood_detection.classifier.train_utils import fit_rf
        clf_full = fit_rf(X_full,y_full)
    elif model in ["mlp","mlp_best_ckpt"]:
        from ood_detection.classifier.train_utils import fit_mlp
        if model == "mlp":
            clf_full = fit_mlp(feature_extractor,X_full,y_full,**kwargs)
        elif model == "mlp_best_ckpt":            
            clf_full = fit_mlp(feature_extractor,X_full,y_full,X_val_ckpt, y_val_ckpt,**kwargs)
    elif model == "adb": 
        from ood_detection.classifier.train_utils import fit_adb       
        clf_full = fit_adb(feature_extractor,X_full,y_full)
    elif model == "mlp_dense_flipout": 
        from ood_detection.classifier.train_utils import fit_mlp_dense_flipout       
        clf_full = fit_mlp_dense_flipout(feature_extractor,X_full,y_full)
    elif model == "biencoder": 
        from ood_detection.classifier.train_utils import fit_biencoder       
        clf_full = fit_biencoder(feature_extractor,df)
    else:
        print("Model's not supported.")
        return

    if not return_metric:
        return clf_full
    else:
        return clf_full,output_metric_dict


def hpo(df,num_trials=200,feature_extractor="xlm",
        sampler="random"):
    from ood_detection.classifier.train_utils import hpo_objective

    if sampler == "random":
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.RandomSampler(seed=0),
                                    pruner=optuna.pruners.HyperbandPruner(reduction_factor=3),
                                   )
    elif sampler == "tpe":
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(multivariate=True,seed=0),
                                    pruner=optuna.pruners.HyperbandPruner(reduction_factor=3),
                                   )
    else:
        print("Sampler's not supported. Supported sampler are: random & tpe")
    study.optimize(lambda trial: hpo_objective(trial, df, feature_extractor), 
                   n_trials=num_trials)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    return trial.params
