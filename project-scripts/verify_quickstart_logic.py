#!/usr/bin/env python3
"""
Quick verification of the ML workflow logic in quickstart.rst
"""

import pandas as pd
import numpy as np

try:
    # Create sample data similar to what users would have
    np.random.seed(42)
    df_converted = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
        'target': np.random.randint(0, 2, 1000)
    })
    print(f"✅ Sample data created: {df_converted.shape}")
    
    # Test the quickstart workflow logic
    import edaflow.ml as ml
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    print("✅ Imports successful")
    
    # Extract features and target (as shown in quickstart)
    X = df_converted.drop('target', axis=1)
    y = df_converted['target']
    print("✅ X, y extraction successful")
    
    # Step 1: Setup ML Experiment - Test DataFrame-style (recommended)
    print("\n🧪 Testing DataFrame-style setup...")
    config = ml.setup_ml_experiment(
        df_converted, 'target',
        val_size=0.15,
        test_size=0.2,
        experiment_name="quick_start_ml",
        random_state=42,
        stratify=True
    )
    print(f"✅ DataFrame-style setup successful, config keys: {list(config.keys())}")
    
    # Alternative: sklearn-style
    print("\n🧪 Testing sklearn-style setup...")
    config_alt = ml.setup_ml_experiment(
        X=X, y=y,
        val_size=0.15,
        test_size=0.2,
        experiment_name="quick_start_ml_alt",
        random_state=42,
        stratify=True
    )
    print(f"✅ Sklearn-style setup successful, config keys: {list(config_alt.keys())}")
    
    # Step 2: Compare Models
    print("\n🤖 Testing model comparison...")
    models = {
        'rf': RandomForestClassifier(random_state=42),
        'lr': LogisticRegression(random_state=42)
    }
    
    # Train models first (as required)
    for name, model in models.items():
        model.fit(config['X_train'], config['y_train'])
        print(f"✅ {name} trained")
    
    # Test compare_models function
    results = ml.compare_models(
        models=models,
        X_train=config['X_train'],
        y_train=config['y_train'],
        X_test=config['X_test'],
        y_test=config['y_test']
    )
    print("✅ Model comparison successful")
    
    # Step 3: Display Results
    print("\n📊 Testing results display...")
    ml.display_leaderboard(results)
    print("✅ Leaderboard display successful")
    
    # Step 4: Optimize Best Model
    print("\n⚡ Testing hyperparameter optimization...")
    tuning_results = ml.optimize_hyperparameters(
        model=RandomForestClassifier(random_state=42),
        X_train=config['X_train'],
        y_train=config['y_train'],
        param_distributions={
            'n_estimators': [50, 100],  # Reduced for speed
            'max_depth': [5, 10]
        },
        method='grid_search',
        verbose=False
    )
    print("✅ Hyperparameter optimization successful")
    
    print("\n" + "="*60)
    print("🎉 QUICKSTART ML WORKFLOW VERIFICATION: SUCCESS!")
    print("="*60)
    print("✅ All steps work correctly")
    print("✅ DataFrame and sklearn-style both work")
    print("✅ Model fitting requirements satisfied")
    print("✅ Function parameters are correct")
    print("✅ No syntax or logic errors found")
    
except Exception as e:
    print(f"\n❌ ERROR in quickstart ML workflow: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*60)
    print("❌ QUICKSTART ML WORKFLOW VERIFICATION: FAILED!")
    print("="*60)
