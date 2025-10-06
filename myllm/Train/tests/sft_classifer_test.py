# examples/train_sft_classifier.py (FULL TEST FUNCTIONALITY - FIXED)
"""
Example for training SFTClassifierTrainer with the new ClassifierConfig
"""

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    from myllm.Train.configs.ClassiferConfig import ClassifierConfig
    from myllm.Configs.ModelConfig import ModelConfig
    from myllm.Train.factory import create_trainer
    from myllm.Train.datasets.toy_dataset import get_toy_dataloader

    print("🚀 Training SFT Classifier with Enhanced Config...")

    # -------------------------
    # Create classifier config
    # -------------------------
    trainer_config = ClassifierConfig(
        # Base model settings
        model_config_name="gpt2-small",
        tokenizer_name="gpt2",
        output_dir="./output_sft_classifier",
        
        # Training parameters
        num_epochs=2,
        batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,  # Base LR
        warmup_steps=10,
        max_grad_norm=1.0,
        eval_steps=5,
        save_steps=10,
        
        # Device settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_compile=False,
        
        # Classification-specific settings
        num_labels=3,
        class_names=["negative", "neutral", "positive"],
        pooling_strategy="last",
        classifier_dropout=0.1,
        classifier_hidden_size=256,
        classifier_learning_rate=5e-4,  # Higher LR for classifier
        base_model_learning_rate=5e-5,  # Lower LR for base model
        
        # Metrics
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        
        # WandB settings
        wandb_project="sft-classifier-enhanced",
        wandb_run_name="gpt2-small-classifier-enhanced",
        wandb_tags=["classification", "enhanced", "unified"],
        report_to=["wandb"],
        load_best_model_at_end=True,
    )

    model_config = ModelConfig.from_name("gpt2-small")

    # -------------------------
    # Create trainer using factory
    # -------------------------
    trainer = create_trainer("sft_classifier", trainer_config, model_config)
    
    # Setup model and tokenizer
    trainer.setup_model()
    print("✅ Model setup complete")
    print(f"✅ Classifier head: {trainer.classifier}")
    print(f"✅ Model device: {trainer.device}")

    # Test tokenizer
    test_text = "This is a sample text for classification"
    encoded = trainer.tokenizer.encode(test_text)
    print(f"✅ Tokenizer test: '{test_text}' -> {encoded}")

    # -------------------------
    # Create dataloaders
    # -------------------------
    train_loader = get_toy_dataloader(
        "classification", 
        batch_size=trainer_config.batch_size,
        tokenizer=trainer.tokenizer,
        num_samples=100,
        max_length=64,
        num_classes=3
    )
    
    eval_loader = get_toy_dataloader(
        "classification",
        batch_size=trainer_config.batch_size,
        tokenizer=trainer.tokenizer,
        num_samples=20,
        max_length=64,
        num_classes=3
    )

    # Test batch
    test_batch = next(iter(train_loader))
    print(f"✅ Batch test: input_ids shape {test_batch['input_ids'].shape}")
    print(f"✅ Batch labels: {test_batch['labels'].tolist()}")
    print(f"✅ Batch texts sample: {test_batch['text'][:2]}")
    print(f"✅ Batch device (before prepare): input_ids={test_batch['input_ids'].device}")

    # Setup data
    trainer.setup_data(train_loader, eval_loader)
    print("✅ Data setup complete")

    # Setup optimizer with differential learning rates
    trainer.setup_optimizer()
    print("✅ Optimizer setup complete")
    print(f"✅ Number of parameter groups: {len(trainer.optimizer.param_groups)}")
    for i, group in enumerate(trainer.optimizer.param_groups):
        print(f"   Group {i}: {group['name']}, LR: {group['lr']:.2e}")

    # -------------------------
    # Comprehensive Testing Suite
    # -------------------------
    print("\n🔍 Running comprehensive diagnostics...")
    
    # Test 1: Debug model outputs
    debug_batch = trainer._prepare_batch(test_batch)
    trainer.debug_model_outputs(debug_batch['input_ids'][:1])

    # Test 2: Test core trainer methods
    print("\n🧪 Testing core trainer functionality...")
    
    test_results = {}
    
    try:
        prepared_batch = trainer._prepare_batch(test_batch)
        test_results['prepare_batch'] = True
        print(f"✅ _prepare_batch works! Device: {prepared_batch['input_ids'].device}")
    except Exception as e:
        test_results['prepare_batch'] = False
        print(f"❌ _prepare_batch failed: {e}")
    
    try:
        step_results = trainer.training_step(prepared_batch)
        test_results['training_step'] = True
        print(f"✅ training_step works! Loss: {step_results['loss']:.4f}, Accuracy: {step_results['accuracy']:.3f}")
    except Exception as e:
        test_results['training_step'] = False
        print(f"❌ training_step failed: {e}")
    
    try:
        eval_results = trainer.evaluation_step(prepared_batch)
        test_results['evaluation_step'] = True
        print(f"✅ evaluation_step works! Loss: {eval_results['loss']:.4f}, Correct: {eval_results['correct']}/{eval_results['total']}")
    except Exception as e:
        test_results['evaluation_step'] = False
        print(f"❌ evaluation_step failed: {e}")

    try:
        eval_metrics = trainer.evaluate()
        test_results['evaluate'] = True
        print(f"✅ evaluate works! Metrics: {eval_metrics}")
    except Exception as e:
        test_results['evaluate'] = False
        print(f"❌ evaluate failed: {e}")

    # Test 3: Test inference functionality - FIXED SECTION
    print("\n🎯 Testing inference capabilities...")
    
    sample_texts = [
        "This is absolutely amazing and wonderful! Great product!",
        "This is okay I guess, not terrible but not great either", 
        "This is terrible and awful, completely disappointed"
    ]
    
    trainer.model.eval()
    trainer.classifier.eval()
    
    inference_results = []
    
    with torch.no_grad():
        for i, text in enumerate(sample_texts):
            try:
                # FIXED: Use batch_encode instead of direct tokenizer call
                inputs = trainer.tokenizer.batch_encode(
                    [text],  # Wrap in list since batch_encode expects list
                    padding=True, 
                    truncation=True, 
                    max_length=64, 
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(trainer.device) for k, v in inputs.items()}
                
                # Get prediction using trainer's methods
                hidden_states = trainer._get_hidden_states(inputs["input_ids"])
                pooled = trainer._pool_hidden_states(hidden_states, inputs.get("attention_mask"))
                logits = trainer.classifier(pooled)
                prediction = torch.argmax(logits, dim=-1).item()
                confidence = torch.softmax(logits, dim=-1).max().item()
                
                result = {
                    'text': text,
                    'prediction': prediction,
                    'class_name': trainer.class_names[prediction],
                    'confidence': confidence,
                    'success': True
                }
                inference_results.append(result)
                
                print(f"📝 Sample {i+1}: '{text[:50]}...'")
                print(f"   → Prediction: {trainer.class_names[prediction]} (confidence: {confidence:.3f})")
                
            except Exception as e:
                result = {
                    'text': text,
                    'error': str(e),
                    'success': False
                }
                inference_results.append(result)
                print(f"❌ Inference failed for sample {i+1}: {e}")

    # Test 4: Test batch inference
    print("\n🔢 Testing batch inference...")
    try:
        # Use a small batch from the dataloader
        test_batch_inference = next(iter(eval_loader))
        prepared_batch_inference = trainer._prepare_batch(test_batch_inference)
        
        with torch.no_grad():
            hidden_states = trainer._get_hidden_states(prepared_batch_inference["input_ids"])
            pooled = trainer._pool_hidden_states(hidden_states, prepared_batch_inference.get("attention_mask"))
            batch_logits = trainer.classifier(pooled)
            batch_predictions = torch.argmax(batch_logits, dim=-1)
            batch_accuracy = (batch_predictions == prepared_batch_inference["labels"]).float().mean().item()
            
            print(f"✅ Batch inference works! Accuracy: {batch_accuracy:.3f}")
            print(f"   Predictions: {batch_predictions.tolist()}")
            print(f"   True labels: {prepared_batch_inference['labels'].tolist()}")
            test_results['batch_inference'] = True
            
    except Exception as e:
        print(f"❌ Batch inference failed: {e}")
        test_results['batch_inference'] = False

    # -------------------------
    # Training Decision
    # -------------------------
    print("\n🎯 Training Decision Analysis...")
    
    critical_tests = ['prepare_batch', 'training_step', 'evaluation_step']
    critical_passed = all(test_results.get(test, False) for test in critical_tests)
    
    optional_tests = ['evaluate', 'batch_inference']
    optional_passed = sum(test_results.get(test, False) for test in optional_tests)
    
    inference_success = sum(1 for r in inference_results if r['success'])
    
    print(f"📊 Test Summary:")
    print(f"   Critical tests: {sum(test_results.get(t, False) for t in critical_tests)}/{len(critical_tests)} passed")
    print(f"   Optional tests: {optional_passed}/{len(optional_tests)} passed") 
    print(f"   Inference samples: {inference_success}/{len(sample_texts)} successful")
    
    if critical_passed:
        print("✅ All critical tests passed! Starting training...")
        
        # Start training
        trainer.train()
        
        print("✅ SFT Classifier training completed!")
        print(f"📁 Output directory: {trainer_config.output_dir}")
        if hasattr(trainer, 'best_checkpoint_path'):
            print(f"🏆 Best checkpoint: {trainer.best_checkpoint_path}")
        if hasattr(trainer, 'best_metric'):
            print(f"📊 Best {trainer_config.metric_for_best_model}: {trainer.best_metric:.4f}")
            
        # Test inference after training - ALSO FIXED
        print("\n🧪 Post-training inference test...")
        trainer.model.eval()
        trainer.classifier.eval()
        
        with torch.no_grad():
            for text in sample_texts:
                try:
                    # FIXED: Use batch_encode for post-training inference too
                    inputs = trainer.tokenizer.batch_encode(
                        [text],  # Wrap in list
                        padding=True, 
                        truncation=True, 
                        max_length=64, 
                        return_tensors="pt"
                    )
                    
                    # Move to device
                    inputs = {k: v.to(trainer.device) for k, v in inputs.items()}
                    
                    hidden_states = trainer._get_hidden_states(inputs["input_ids"])
                    pooled = trainer._pool_hidden_states(hidden_states, inputs.get("attention_mask"))
                    logits = trainer.classifier(pooled)
                    prediction = torch.argmax(logits, dim=-1).item()
                    confidence = torch.softmax(logits, dim=-1).max().item()
                    
                    print(f"🎯 '{text[:40]}...' -> {trainer.class_names[prediction]} (confidence: {confidence:.3f})")
                    
                except Exception as e:
                    print(f"❌ Post-training inference failed: {e}")
                    
    else:
        print("❌ Critical tests failed. Skipping training to debug issues.")
        print("💡 Failed tests:")
        for test in critical_tests:
            if not test_results.get(test, False):
                print(f"   - {test}")
        
        print("\n🔧 Debugging suggestions:")
        if not test_results.get('prepare_batch', False):
            print("   - Check device compatibility and tensor types in _prepare_batch")
        if not test_results.get('training_step', False):
            print("   - Verify model output shapes and loss computation")
        if not test_results.get('evaluation_step', False):
            print("   - Check evaluation mode and gradient handling")

    print("\n🎉 Test suite completed!")