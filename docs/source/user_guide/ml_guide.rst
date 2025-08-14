Machine Learning User Guide
===========================

This guide provides comprehensive examples and workflows for using edaflow's ML functions effectively.

Overview
--------

The edaflow.ml subpackage provides 26 functions organized into 5 categories:

* **Configuration & Setup** (3 functions): Experiment setup and data validation
* **Model Comparison** (4 functions): Multi-model evaluation and ranking  
* **Hyperparameter Tuning** (4 functions): Optimization strategies
* **Performance Visualization** (6 functions): ML-specific plots and curves
* **Model Artifacts** (4 functions): Model persistence and experiment tracking

Best Practice Scoring Metrics
---------------------------


Choosing the right scoring metric is critical for evaluating and comparing machine learning models. Here are the best practice metrics supported by edaflow, with practical guidance:

**When to Choose Accuracy or F1 as Your Primary Metric**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Selecting a primary metric depends on your dataset and business goals:

- **Accuracy** is best when your classes are balanced and all types of errors are equally important. It measures the overall proportion of correct predictions. Use accuracy as your primary metric when:
    - The dataset has roughly equal numbers of samples in each class.
    - False positives and false negatives have similar costs.
    - Example: Handwritten digit recognition, animal type classification with balanced classes.

- **F1 Score** is best when your classes are imbalanced or when you care about both precision and recall. It is especially useful when the positive class is rare or when missing positive cases is costly. Use F1 as your primary metric when:
    - The dataset is imbalanced (one class is much less frequent).
    - Both false positives and false negatives are important to minimize.
    - Example: Disease detection, fraud detection, spam filtering.

**Summary:**
- Use **accuracy** for balanced datasets and equal error costs.
- Use **F1** for imbalanced datasets or when both precision and recall matter.

Tracking both metrics can provide a more complete picture, but always select a primary metric that aligns with your real-world goals.

**Classification Metrics:**

- **accuracy**: Overall correctness. Use for balanced datasets.
- **precision**: Correctness of positive predictions. Important for imbalanced data (e.g., fraud detection).
- **recall**: Ability to find all positive samples. Use when missing positives is costly (e.g., medical diagnosis).
- **f1**: Harmonic mean of precision and recall. Best for imbalanced data when both precision and recall matter.
- **roc_auc**: Area under the ROC curve. Measures ranking quality, best for binary classification.

**Regression Metrics:**

- **mse**: Mean squared error. Penalizes large errors, sensitive to outliers.
- **mae**: Mean absolute error. Robust to outliers, interpretable.
- **rmse**: Root mean squared error. Like MSE, but in original units.
- **r2**: R-squared. Proportion of variance explained by the model.

**How to Use in edaflow**


You can specify any of these metrics in the `scoring` argument of `ml.compare_models`.

**About cv_folds**
~~~~~~~~~~~~~~~~~
The `cv_folds` parameter controls the number of cross-validation folds used to evaluate each model. Cross-validation splits your training data into several parts (folds), trains the model on some folds, and validates it on the remaining fold, repeating this process for each fold. The results are averaged to give a more reliable estimate of model performance.

- Typical values: 5 or 10 (5 is common and a good default)
- More folds = more reliable estimates, but longer runtime
- Use higher values for small datasets, and lower values for very large datasets

Example: `cv_folds=5` means 5-fold cross-validation (the data is split into 5 parts, each used once as validation).

.. code-block:: python

   # Example: Compare models using all best practice metrics
   results = ml.compare_models(
       models=models,
       X_train=config['X_train'],
       y_train=config['y_train'],
       X_test=config['X_test'],
       y_test=config['y_test'],
       cv_folds=5,
       scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
   )

   # For regression:
   results = ml.compare_models(
       models=models,
       X_train=X_train,
       y_train=y_train,
       X_test=X_test,
       y_test=y_test,
       cv_folds=5,
       scoring=['mse', 'mae', 'rmse', 'r2']
   )

**Tip:**
- For imbalanced classification, prefer `f1`, `precision`, and `recall` over `accuracy`.
- For regression, use both `mae` and `rmse` to understand error characteristics.

These metrics are recommended for most practical ML workflows and are fully supported in edaflow.

Choosing Metrics by Problem Type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The best metric depends on your prediction target:

**Binary Classification (2 classes):**
- Use: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`
- `roc_auc` is only available for binary targets (e.g., 0/1, True/False).
- Example: Disease prediction (yes/no), fraud detection (fraud/not fraud)

**Multiclass Classification (3+ classes):**
- Use: `accuracy`, `precision`, `recall`, `f1`
- `roc_auc` is not available in edaflow for multiclass (will show NaN)
- Example: Animal type (cat/dog/horse), digit recognition (0-9)

**Regression (continuous target):**
- Use: `mse`, `mae`, `rmse`, `r2`
- Example: House price prediction, temperature forecasting

**Tip:**
- If you see NaN for `roc_auc`, check if your target is multiclass or if your model lacks probability outputs.
- For multiclass ROC AUC, use scikit-learn directly or request an edaflow extension.

This guidance ensures you always choose the right metric for your ML problem type.

Practical Examples: Metric Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------------+---------------------+-------------------------------+
| Scenario                  | Recommended Metrics | Why/When to Use               |
+===========================+=====================+===============================+
| Disease prediction        | f1, recall, roc_auc | Imbalanced, missing positives |
| (binary classification)   |                     | is costly                     |
+---------------------------+---------------------+-------------------------------+
| Spam detection            | precision, f1       | Imbalanced, false positives   |
| (binary classification)   |                     | are costly                    |
+---------------------------+---------------------+-------------------------------+
| Animal type classification| accuracy, f1        | Multiclass, balanced classes  |
| (multiclass classification)|                    |                               |
+---------------------------+---------------------+-------------------------------+
| Digit recognition         | accuracy, f1        | Multiclass, balanced          |
| (multiclass classification)|                    |                               |
+---------------------------+---------------------+-------------------------------+
| House price prediction    | mae, rmse, r2       | Regression, interpretability  |
| (regression)              |                     | and error size matter         |
+---------------------------+---------------------+-------------------------------+
| Energy demand forecasting | mae, mse, r2        | Regression, outlier-robust    |
| (regression)              |                     | and variance explained        |
+---------------------------+---------------------+-------------------------------+

**Tip:**
- For imbalanced binary classification, use `f1`, `recall`, and `roc_auc`.
- For multiclass, use `accuracy` and `f1`.
- For regression, use both `mae` and `rmse` for a full error picture.

Complete ML Workflow Example
-----------------------------

Here's a comprehensive example showing the full ML workflow:

.. code-block:: python

   import edaflow.ml as ml
   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
   from sklearn.linear_model import LogisticRegression
   from sklearn.svm import SVC

   # Load your data
   df = pd.read_csv('your_data.csv')
   X = df.drop('target', axis=1)
   y = df['target']

   # Step 1: Setup ML Experiment
   config = ml.setup_ml_experiment(
       X=X, 
       y=y,
       test_size=0.2,
       val_size=0.15,
       experiment_name="comprehensive_model_comparison",
       random_state=42
   )

   # Step 2: Validate Data Quality
   validation_report = ml.validate_ml_data(
       X=config['X_train'],
       y=config['y_train'],
       check_missing=True,
       check_cardinality=True,
       check_distributions=True
   )

   # Step 3: Configure Preprocessing Pipeline
   pipeline_config = ml.configure_model_pipeline(
       data_config=config,
       numerical_strategy='standard',
       categorical_strategy='onehot',
       handle_missing='impute',
       verbose=True
   )

   # Step 4: Compare Multiple Models
   models = {
       'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
       'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
       'logistic_regression': LogisticRegression(random_state=42),
       'svm': SVC(probability=True, random_state=42)
   }

   # 🚨 CRITICAL: Train all models first!
   print("🔧 Training models...")
   for name, model in models.items():
       model.fit(config['X_train'], config['y_train'])
       print(f"✅ {name} trained")

   comparison_results = ml.compare_models(
       models=models,
       X_train=config['X_train'],
       y_train=config['y_train'],
       X_test=config['X_test'],
       y_test=config['y_test'],
       cv_folds=5,
       scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
   )

   # Step 5: Display Model Leaderboard
   ml.display_leaderboard(
       comparison_results=comparison_results,
       sort_by='roc_auc',
       ascending=False,
       show_std=True,
       figsize=(12, 4)
   )

   # Step 6: Rank Models and Select Best Performer
   # Two ways to get the best model:
   
   # Method 1: DataFrame format (traditional)
   ranked_df = ml.rank_models(comparison_results, 'roc_auc')
   best_model_traditional = ranked_df.iloc[0]['model']
   
   # Method 2: List format (easy dictionary access)
   best_model = ml.rank_models(
       comparison_results, 
       'roc_auc', 
       return_format='list'
   )[0]['model_name']
   
   print(f"Best performing model: {best_model}")
   
   # Step 7: Hyperparameter Optimization for Best Model
   if best_model == 'random_forest':
       param_distributions = {
           'n_estimators': [50, 100, 200],
           'max_depth': [3, 5, 7, None],
           'min_samples_split': [2, 5, 10],
           'min_samples_leaf': [1, 2, 4]
       }
   
   tuning_results = ml.optimize_hyperparameters(
       model=RandomForestClassifier(random_state=42),
       X_train=config['X_train'],
       y_train=config['y_train'],
       param_distributions=param_distributions,
       method='random',
       n_iter=50,
       cv=5,
       scoring='roc_auc'
   )

   # Step 8: Performance Visualizations
   best_tuned_model = tuning_results['best_model']
   
   # Learning curves
   ml.plot_learning_curves(
       model=best_tuned_model,
       X_train=config['X_train'],
       y_train=config['y_train'],
       cv=5,
       scoring='roc_auc'
   )
   
   # ROC curves
   ml.plot_roc_curves(
       models={'tuned_model': best_tuned_model},
       X_val=config['X_test'],
       y_val=config['y_test']
   )
   
   # Feature importance
   ml.plot_feature_importance(
       model=best_tuned_model,
       feature_names=config['X_train'].columns,
       top_n=15
   )

   # Step 9: Save Model Artifacts
   artifact_paths = ml.save_model_artifacts(
       model=best_tuned_model,
       model_name="best_tuned_rf_model",
       experiment_config=config,
       performance_metrics=tuning_results['best_score_dict'],
       save_dir="production_models",
       include_data_sample=True,
       X_sample=config['X_train'].head(100)
   )

   # Step 10: Track Experiment
   ml.track_experiment(
       experiment_name=config['experiment_name'],
       model_results=comparison_results,
       tuning_results=tuning_results,
       final_model_path=artifact_paths['model_path'],
       notes="Comprehensive model comparison with hyperparameter tuning"
   )

   # Step 11: Generate Model Report
   ml.create_model_report(
       model=best_tuned_model,
       experiment_config=config,
       performance_metrics=tuning_results['best_score_dict'],
       model_comparison=comparison_results,
       save_path="model_reports/comprehensive_analysis.pdf"
   )

Individual Function Examples
----------------------------

Configuration Functions
~~~~~~~~~~~~~~~~~~~~~~~~

**Setup ML Experiment**

.. code-block:: python

   # Basic setup
   config = ml.setup_ml_experiment(X=X, y=y)
   
   # Advanced setup with custom splits
   config = ml.setup_ml_experiment(
       X=X, y=y,
       test_size=0.2,
       val_size=0.15,
       stratify=True,
       experiment_name="advanced_experiment",
       random_state=42,
       create_directories=True
   )

**Validate ML Data**

.. code-block:: python

   # Comprehensive data validation
   report = ml.validate_ml_data(
       X=X_train, y=y_train,
       check_missing=True,
       check_cardinality=True,
       check_distributions=True,
       missing_threshold=0.1,
       high_cardinality_threshold=50
   )

Model Comparison Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Compare Models**

.. code-block:: python

   # Quick model comparison
   models = {
       'rf': RandomForestClassifier(),
       'lr': LogisticRegression(),
       'svm': SVC(probability=True)
   }
   
   results = ml.compare_models(
       models=models,
       X_train=X_train, y_train=y_train,
       X_test=X_test, y_test=y_test,
       cv_folds=5
   )

**Display Leaderboard**

.. code-block:: python

   # Show model rankings
   ml.display_leaderboard(
       comparison_results=results,
       sort_by='f1_score',
       show_std=True,
       highlight_best=True,
       figsize=(12, 4)
   )

**Rank Models**

The ``rank_models`` function provides flexible model ranking with two return formats:

.. code-block:: python

   # DataFrame format (traditional, backward compatible)
   ranked_df = ml.rank_models(
       comparison_df=results,
       primary_metric='accuracy'
   )
   
   # Access best model
   best_model = ranked_df.iloc[0]['model']
   best_accuracy = ranked_df.iloc[0]['accuracy']
   
   print(f"Best model: {best_model} (accuracy: {best_accuracy:.4f})")

   # List format (dictionary access)
   ranked_list = ml.rank_models(
       comparison_df=results,
       primary_metric='accuracy',
       return_format='list'
   )
   
   # Easy dictionary access patterns
   best_model_name = ranked_list[0]["model_name"]
   best_accuracy = ranked_list[0]["accuracy"]
   best_f1 = ranked_list[0]["f1"]
   
   # One-liner pattern for best model
   best_model = ml.rank_models(results, 'accuracy', return_format='list')[0]["model_name"]
   
   # Access all ranked models
   print("All models ranked by accuracy:")
   for i, model_info in enumerate(ranked_list):
       print(f"{i+1}. {model_info['model_name']}: {model_info['accuracy']:.4f}")

**Advanced Ranking Options**

.. code-block:: python

   # Rank by different metrics
   ranked_by_f1 = ml.rank_models(results, 'f1_score', return_format='list')
   ranked_by_precision = ml.rank_models(results, 'precision', return_format='list')
   
   # Ascending order (useful for error metrics)
   ranked_by_error = ml.rank_models(
       results, 
       'validation_error', 
       ascending=True,  # Lower error is better
       return_format='list'
   )
   
   # Weighted multi-metric ranking
   ranked_weighted = ml.rank_models(
       comparison_df=results,
       primary_metric='accuracy',
       weights={
           'accuracy': 0.4,
           'f1_score': 0.3,
           'precision': 0.2,
           'recall': 0.1
       },
       return_format='list'
   )
   
   best_overall = ranked_weighted[0]["model_name"]
   print(f"Best model by weighted score: {best_overall}")

**Return Format Comparison**

.. code-block:: python

   # Both formats provide the same ranking
   df_format = ml.rank_models(results, 'accuracy')
   list_format = ml.rank_models(results, 'accuracy', return_format='list')
   
   # DataFrame format - good for analysis and display
   print("Top 3 models (DataFrame):")
   print(df_format.head(3)[['model', 'accuracy', 'f1', 'rank']])
   
   # List format - easy programmatic access
   print("Top 3 models (List):")
   for i, model in enumerate(list_format[:3]):
       print(f"{i+1}. {model['model_name']}: {model['accuracy']:.4f}")
   
   # Choose format based on your needs:
   # - DataFrame: Analysis, filtering, display
   # - List: Simple access, iteration, one-liners

Hyperparameter Tuning Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Grid Search**

.. code-block:: python

   param_grid = {
       'n_estimators': [100, 200],
       'max_depth': [3, 5, None]
   }
   
   grid_results = ml.grid_search_models(
       models={'RandomForest': RandomForestClassifier()},
       param_grids={'RandomForest': param_grid},
       X_train=X_train, y_train=y_train,
       cv=5,
       scoring='accuracy'
   )

**Bayesian Optimization**

.. code-block:: python

   param_space = {
       'n_estimators': (50, 200),
       'max_depth': (3, 10),
       'min_samples_split': (2, 20)
   }
   
   bayes_results = ml.bayesian_optimization(
       model=RandomForestClassifier(),
       param_space=param_space,
       X_train=X_train, y_train=y_train,
       n_calls=50,
       cv=5
   )

Performance Visualization Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Learning Curves**

.. code-block:: python

   ml.plot_learning_curves(
       model=model,
       X_train=X_train, y_train=y_train,
       cv=5,
       train_sizes=np.linspace(0.1, 1.0, 10),
       scoring='f1_weighted'
   )

**ROC Curves**

.. code-block:: python

   ml.plot_roc_curves(
       models={'Model 1': model1, 'Model 2': model2},
       X_val=X_test, y_val=y_test,
       title="Model Comparison ROC Curves"
   )

Model Artifacts Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

**Save Model Artifacts**

.. code-block:: python

   paths = ml.save_model_artifacts(
       model=trained_model,
       model_name="production_model_v1",
       experiment_config=config,
       performance_metrics=metrics,
       save_dir="models/production",
       format='joblib'
   )

**Load Model Artifacts**

.. code-block:: python

   loaded_artifacts = ml.load_model_artifacts(
       model_path="models/production/production_model_v1.joblib"
   )
   
   model = loaded_artifacts['model']
   config = loaded_artifacts['config']
   metrics = loaded_artifacts['metrics']

Best Practices
--------------

1. **Always start with setup_ml_experiment()** to ensure consistent data splits
2. **Validate your data** with validate_ml_data() before training
3. **Use compare_models()** to evaluate multiple algorithms quickly  
4. **Apply hyperparameter tuning** only to your best-performing models
5. **Save model artifacts** with comprehensive metadata for reproducibility
6. **Track experiments** to maintain a history of your ML work
7. **Generate model reports** for stakeholder communication

Integration with EDA
---------------------

The ML functions integrate seamlessly with edaflow's EDA capabilities:

.. code-block:: python

   # Start with EDA
   edaflow.check_null_columns(df)
   edaflow.analyze_categorical_columns(df) 
   edaflow.visualize_heatmap(df)
   
   # Clean and prepare data
   df_clean = edaflow.convert_to_numeric(df)
   df_imputed = edaflow.impute_numerical_median(df_clean)
   
   # Transition to ML workflow  
   X = df_imputed.drop('target', axis=1)
   y = df_imputed['target']
   
   config = ml.setup_ml_experiment(X=X, y=y)
   # ... continue with ML workflow

This creates a complete data science pipeline from exploration to model deployment.
