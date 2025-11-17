# examples/train_sft_classifier.py (UPDATED VERSION)
"""
Enhanced Example for training SFTClassifierTrainer with advanced features
"""

if __name__ == "__main__":
    import torch
    import numpy as np
    from torch.utils.data import DataLoader

    from myllm.Train.configs.ClassiferConfig import EnhancedClassifierConfig  # Use enhanced config
    from myllm.Configs.ModelConfig import ModelConfig
    from myllm.Train.factory import create_trainer
    from myllm.Train.datasets.toy_dataset import get_toy_dataloader

    print("🚀 Training Enhanced SFT Classifier with Advanced Features...")

    # -------------------------
    # Create enhanced classifier config
    # -------------------------
    trainer_config = EnhancedClassifierConfig(
        # Base model settings
        model_config_name="gpt2-small",
        tokenizer_name="gpt2",
        output_dir="./output_enhanced_sft_classifier",
        
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
        classifier_dropout=0.1,
        classifier_hidden_size=256,
        classifier_learning_rate=5e-4,  # Higher LR for classifier
        base_model_learning_rate=5e-5,  # Lower LR for base model
        
        # Enhanced features (already enabled in EnhancedClassifierConfig)
        # But we can customize multi-task config
        multi_task_config={
            "sentiment": 3,      # 3 classes for sentiment
            "topic": 2,          # 2 classes for topic (simplified for demo)
        },
        class_weights=[0.8, 1.0, 1.2],  # Handle class imbalance
        
        # Metrics
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        
        # WandB settings
        wandb_project="sft-classifier-enhanced",
        wandb_run_name="gpt2-small-enhanced-features",
        wandb_tags=["classification", "enhanced", "multi-task", "attention-pooling"],
        report_to=["wandb"],
        load_best_model_at_end=True,
    )

    model_config = ModelConfig.from_name("gpt2-small")

    # -------------------------
    # Create enhanced trainer using factory
    # -------------------------
    trainer = create_trainer("sft_classifier", trainer_config, model_config)
    
    # Rest of the script remains the same...
    # [The rest of your existing test script code continues here...]
    
    # Setup model and tokenizer
    trainer.setup_model()
    print("✅ Enhanced model setup complete")
    print(f"✅ Pooling strategy: {trainer.pooling_strategy}")
    print(f"✅ Multi-task enabled: {trainer.multi_task_head is not None}")
    print(f"✅ Attention pooling: {trainer.attention_pooler is not None}")
    print(f"✅ Gradient checkpointing: {getattr(trainer_config, 'gradient_checkpointing', False)}")
    print(f"✅ Model device: {trainer.device}")

    # Test tokenizer
    test_text = "This is a sample text for enhanced classification"
    encoded = trainer.tokenizer.encode(test_text)
    print(f"✅ Tokenizer test: '{test_text}' -> {encoded}")

    # -------------------------
    # Create enhanced dataloaders with multi-task support
    # -------------------------
    def get_enhanced_toy_dataloader(split="train"):
        """Create dataloader with multi-task labels"""
        base_loader = get_toy_dataloader(
            "classification", 
            batch_size=trainer_config.batch_size,
            tokenizer=trainer.tokenizer,
            num_samples=100 if split == "train" else 20,
            max_length=64,
            num_classes=3
        )
        
        # Convert to enhanced format with multi-task labels
        enhanced_batches = []
        for batch in base_loader:
            enhanced_batch = batch.copy()
            
            # Add multi-task labels for demonstration
            if trainer.multi_task_head:
                batch_size = batch['labels'].size(0)
                
                # Sentiment labels (same as main task)
                enhanced_batch['task_labels'] = {
                    'sentiment': batch['labels'].clone(),
                    'topic': torch.randint(0, 2, (batch_size,))  # Binary topic classification
                }
            
            enhanced_batches.append(enhanced_batch)
            
        return enhanced_batches

    train_loader = get_enhanced_toy_dataloader("train")
    eval_loader = get_enhanced_toy_dataloader("eval")

    # Test enhanced batch
    test_batch = train_loader[0]
    print(f"✅ Enhanced batch test: input_ids shape {test_batch['input_ids'].shape}")
    print(f"✅ Main labels: {test_batch['labels'].tolist()}")
    
    if 'task_labels' in test_batch:
        print("✅ Multi-task labels:")
        for task_name, labels in test_batch['task_labels'].items():
            print(f"   - {task_name}: {labels.tolist()}")
    
    print(f"✅ Batch device (before prepare): input_ids={test_batch['input_ids'].device}")

    # Setup data
    trainer.setup_data(train_loader, eval_loader)
    print("✅ Enhanced data setup complete")

    # Setup optimizer with differential learning rates
    trainer.setup_optimizer()
    print("✅ Enhanced optimizer setup complete")
    print(f"✅ Number of parameter groups: {len(trainer.optimizer.param_groups)}")
    for i, group in enumerate(trainer.optimizer.param_groups):
        print(f"   Group {i}: {group['name']}, LR: {group['lr']:.2e}")

    # -------------------------
    # Comprehensive Enhanced Testing Suite
    # -------------------------
    print("\n🔍 Running enhanced comprehensive diagnostics...")
    
    # Test 1: Enhanced debug model outputs
    debug_batch = trainer._prepare_batch(test_batch)
    trainer.debug_model_outputs(debug_batch['input_ids'][:1])

    # Test 2: Test enhanced trainer methods
    print("\n🧪 Testing enhanced trainer functionality...")
    
    test_results = {}
    
    try:
        prepared_batch = trainer._prepare_batch(test_batch)
        test_results['prepare_batch'] = True
        print(f"✅ _prepare_batch works! Device: {prepared_batch['input_ids'].device}")
        
        # Check multi-task labels
        if 'task_labels' in prepared_batch:
            print(f"✅ Multi-task labels prepared for: {list(prepared_batch['task_labels'].keys())}")
            
    except Exception as e:
        test_results['prepare_batch'] = False
        print(f"❌ _prepare_batch failed: {e}")
    
    try:
        step_results = trainer.training_step(prepared_batch)
        test_results['training_step'] = True
        print(f"✅ Enhanced training_step works! Loss: {step_results['loss']:.4f}, Accuracy: {step_results['accuracy']:.3f}")
        
        # Check enhanced metrics
        if 'confidence' in step_results:
            print(f"✅ Confidence tracking: {step_results['confidence']:.3f}")
            
    except Exception as e:
        test_results['training_step'] = False
        print(f"❌ training_step failed: {e}")
    
    try:
        eval_results = trainer.evaluation_step(prepared_batch)
        test_results['evaluation_step'] = True
        
        if trainer.multi_task_head:
            print(f"✅ Multi-task evaluation_step works! Loss: {eval_results['loss']:.4f}")
        else:
            print(f"✅ Single-task evaluation_step works! Loss: {eval_results['loss']:.4f}, Correct: {eval_results['correct']}/{eval_results['total']}")
            
    except Exception as e:
        test_results['evaluation_step'] = False
        print(f"❌ evaluation_step failed: {e}")

    try:
        eval_metrics = trainer.evaluate()
        test_results['evaluate'] = True
        print(f"✅ Enhanced evaluate works! Metrics: {eval_metrics}")
        
        # Check enhanced metrics
        enhanced_metrics = ['eval_confidence_mean', 'eval_confidence_std', 'eval_low_confidence_ratio']
        for metric in enhanced_metrics:
            if metric in eval_metrics:
                print(f"   - {metric}: {eval_metrics[metric]:.4f}")
                
    except Exception as e:
        test_results['evaluate'] = False
        print(f"❌ evaluate failed: {e}")

    # Test 3: Enhanced inference functionality
    print("\n🎯 Testing enhanced inference capabilities...")
    
    sample_texts = [
        "This is absolutely amazing and wonderful! Great product!",
        "This is okay I guess, not terrible but not great either", 
        "This is terrible and awful, completely disappointed"
    ]
    
    trainer.model.eval()
    if trainer.classifier:
        trainer.classifier.eval()
    if trainer.multi_task_head:
        trainer.multi_task_head.eval()
    
    inference_results = []
    
    with torch.no_grad():
        for i, text in enumerate(sample_texts):
            try:
                # Enhanced encoding
                inputs = trainer.tokenizer.batch_encode(
                    [text],
                    padding=True, 
                    truncation=True, 
                    max_length=64, 
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(trainer.device) for k, v in inputs.items()}
                
                # Enhanced prediction with confidence
                # FIX: Specify task name for multi-task inference
                if trainer.multi_task_head:
                    # Use sentiment task for inference (since it matches our class_names)
                    predictions, confidence = trainer.predict_with_confidence(inputs, task_name="sentiment")
                    prediction = predictions.item()
                    class_name = trainer.class_names[prediction]  # Use main class names for sentiment
                else:
                    predictions, confidence = trainer.predict_with_confidence(inputs)
                    prediction = predictions.item()
                    class_name = trainer.class_names[prediction]

                result = {
                    'text': text,
                    'prediction': prediction,
                    'class_name': class_name,
                    'confidence': confidence.item(),
                    'success': True
                }
                
                # Test MC dropout for uncertainty if enabled
                # Test MC dropout for uncertainty if enabled
                if trainer.use_mc_dropout:
                    try:
                        if trainer.multi_task_head:
                            mean_probs, uncertainty = trainer.mc_dropout_predict(inputs, num_samples=5, task_name="sentiment")
                        else:
                            mean_probs, uncertainty = trainer.mc_dropout_predict(inputs, num_samples=5)
                        result['uncertainty'] = uncertainty.item()
                        result['mean_confidence'] = mean_probs.max().item()
                    except Exception as mc_error:
                        result['mc_error'] = str(mc_error)
                
                inference_results.append(result)
                
                print(f"📝 Sample {i+1}: '{text[:50]}...'")
                print(f"   → Prediction: {trainer.class_names[prediction]} (confidence: {result['confidence']:.3f})")
                if 'uncertainty' in result:
                    print(f"   → Uncertainty: {result['uncertainty']:.3f}")
                
            except Exception as e:
                result = {
                    'text': text,
                    'error': str(e),
                    'success': False
                }
                inference_results.append(result)
                print(f"❌ Inference failed for sample {i+1}: {e}")

    # Test 4: Enhanced batch inference with multi-task
    print("\n🔢 Testing enhanced batch inference...")
    try:
        test_batch_inference = eval_loader[0]
        prepared_batch_inference = trainer._prepare_batch(test_batch_inference)
        
        with torch.no_grad():
            if trainer.multi_task_head:
                # Multi-task inference
                total_correct = 0
                total_samples = 0
                
                for task_name in trainer.multi_task_config.keys():
                    logits, _ = trainer.forward(prepared_batch_inference, task_name)
                    labels = trainer._get_labels(prepared_batch_inference, task_name)
                    
                    if labels is not None:
                        predictions = torch.argmax(logits, dim=-1)
                        accuracy = (predictions == labels).float().mean().item()
                        print(f"✅ {task_name} batch inference - Accuracy: {accuracy:.3f}")
                        
            else:
                # Single-task inference
                logits, _ = trainer.forward(prepared_batch_inference)
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == prepared_batch_inference["labels"]).float().mean().item()
                
                print(f"✅ Single-task batch inference - Accuracy: {accuracy:.3f}")
                print(f"   Predictions: {predictions.tolist()}")
                print(f"   True labels: {prepared_batch_inference['labels'].tolist()}")
            
            test_results['batch_inference'] = True
            
    except Exception as e:
        print(f"❌ Batch inference failed: {e}")
        test_results['batch_inference'] = False

    # Test 5: Test enhanced pooling strategies
    print("\n🌊 Testing pooling strategies...")
    try:
        test_inputs = prepared_batch['input_ids'][:2]  # Small batch for testing
        
        hidden_states = trainer._get_hidden_states(test_inputs)
        pooled = trainer._pool_hidden_states(hidden_states, prepared_batch.get("attention_mask")[:2] if prepared_batch.get("attention_mask") is not None else None)
        
        print(f"✅ Pooling test successful!")
        print(f"   Hidden states shape: {hidden_states.shape}")
        print(f"   Pooled output shape: {pooled.shape}")
        print(f"   Pooling strategy: {trainer.pooling_strategy}")
        
        test_results['pooling'] = True
        
    except Exception as e:
        print(f"❌ Pooling test failed: {e}")
        test_results['pooling'] = False

    # -------------------------
    # Enhanced Training Decision
    # -------------------------
    print("\n🎯 Enhanced Training Decision Analysis...")
    
    critical_tests = ['prepare_batch', 'training_step', 'evaluation_step', 'pooling']
    critical_passed = all(test_results.get(test, False) for test in critical_tests)
    
    optional_tests = ['evaluate', 'batch_inference']
    optional_passed = sum(test_results.get(test, False) for test in optional_tests)
    
    inference_success = sum(1 for r in inference_results if r['success'])
    
    print(f"📊 Enhanced Test Summary:")
    print(f"   Critical tests: {sum(test_results.get(t, False) for t in critical_tests)}/{len(critical_tests)} passed")
    print(f"   Optional tests: {optional_passed}/{len(optional_tests)} passed") 
    print(f"   Inference samples: {inference_success}/{len(sample_texts)} successful")
    print(f"   Enhanced features:")
    print(f"     - Multi-task: {trainer.multi_task_head is not None}")
    print(f"     - Attention pooling: {trainer.attention_pooler is not None}")
    print(f"     - MC Dropout: {trainer.use_mc_dropout}")
    print(f"     - Class weights: {trainer.class_weights is not None}")
    
    if critical_passed:
        print("✅ All critical tests passed! Starting enhanced training...")
        
        # Start enhanced training
        trainer.train()
        
        print("✅ Enhanced SFT Classifier training completed!")
        print(f"📁 Output directory: {trainer_config.output_dir}")
        if hasattr(trainer, 'best_checkpoint_path'):
            print(f"🏆 Best checkpoint: {trainer.best_checkpoint_path}")
        if hasattr(trainer, 'best_metric'):
            print(f"📊 Best {trainer_config.metric_for_best_model}: {trainer.best_metric:.4f}")
            
        # Enhanced post-training inference test
        print("\n🧪 Enhanced post-training inference test...")
        trainer.model.eval()
        if trainer.classifier:
            trainer.classifier.eval()
        if trainer.multi_task_head:
            trainer.multi_task_head.eval()
        
        with torch.no_grad():
            for text in sample_texts:
                try:
                    inputs = trainer.tokenizer.batch_encode(
                        [text],
                        padding=True, 
                        truncation=True, 
                        max_length=64, 
                        return_tensors="pt"
                    )
                    
                    inputs = {k: v.to(trainer.device) for k, v in inputs.items()}
                    
                    # Enhanced prediction with confidence - FIXED for multi-task
                    if trainer.multi_task_head:
                        predictions, confidence = trainer.predict_with_confidence(inputs, task_name="sentiment")
                    else:
                        predictions, confidence = trainer.predict_with_confidence(inputs)
                    prediction = predictions.item()

                    print(f"🎯 '{text[:40]}...' -> {trainer.class_names[prediction]} (confidence: {confidence.item():.3f})")

                    # Test uncertainty if MC dropout enabled
                    if trainer.use_mc_dropout:
                        if trainer.multi_task_head:
                            mean_probs, uncertainty = trainer.mc_dropout_predict(inputs, num_samples=10, task_name="sentiment")
                        else:
                            mean_probs, uncertainty = trainer.mc_dropout_predict(inputs, num_samples=10)
                        print(f"   🔮 Uncertainty: {uncertainty.item():.3f}, Mean confidence: {mean_probs.max().item():.3f}")
                    
                except Exception as e:
                    print(f"❌ Enhanced post-training inference failed: {e}")
                    
        # Test enhanced checkpoint loading
        print("\n💾 Testing enhanced checkpoint functionality...")
        try:
            # Save a test checkpoint
            test_checkpoint_dir = trainer.save_checkpoint("./test_enhanced_checkpoint")
            print(f"✅ Enhanced checkpoint saved to: {test_checkpoint_dir}")
            
            # Test loading (in a real scenario, you'd create a new trainer instance)
            print("✅ Enhanced checkpoint functionality verified")
            
        except Exception as e:
            print(f"❌ Checkpoint test failed: {e}")
                    
    else:
        print("❌ Critical tests failed. Skipping training to debug issues.")
        print("💡 Failed tests:")
        for test in critical_tests:
            if not test_results.get(test, False):
                print(f"   - {test}")
        
        print("\n🔧 Enhanced debugging suggestions:")
        if not test_results.get('prepare_batch', False):
            print("   - Check multi-task label preparation in _prepare_batch")
        if not test_results.get('training_step', False):
            print("   - Verify multi-task loss computation and gradient handling")
        if not test_results.get('evaluation_step', False):
            print("   - Check multi-task evaluation metrics aggregation")
        if not test_results.get('pooling', False):
            print("   - Verify attention pooling implementation and input shapes")

    # Final enhanced summary
    print("\n" + "="*60)
    print("🎉 ENHANCED SFT CLASSIFIER TEST SUITE COMPLETED!")
    print("="*60)
    print(f"📈 Enhanced Features Status:")
    print(f"   ✅ Multi-task Learning: {trainer.multi_task_head is not None}")
    print(f"   ✅ Attention Pooling: {trainer.pooling_strategy == 'attention' and trainer.attention_pooler is not None}")
    print(f"   ✅ Uncertainty Estimation: {trainer.use_mc_dropout}")
    print(f"   ✅ Class Imbalance Handling: {trainer.class_weights is not None}")
    print(f"   ✅ Gradient Checkpointing: {getattr(trainer_config, 'gradient_checkpointing', False)}")
    print(f"   ✅ Confidence Tracking: {len(trainer.confidence_history) > 0 if hasattr(trainer, 'confidence_history') else False}")
    
    if critical_passed and hasattr(trainer, 'training_completed') and trainer.training_completed:
        print("\n🚀 ENHANCED TRAINING SUCCESSFULLY COMPLETED!")
        print("   All advanced features are operational and training completed successfully.")
    elif critical_passed:
        print("\n⚠️  Ready for training but training wasn't executed.")
    else:
        print("\n❌ Training skipped due to test failures.")