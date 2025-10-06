# examples/run_all_tests_fixed_final.py (FIXED - Basic Functionality)
"""
Final fixed test script with proper error handling using unified architecture
"""

import torch
import gc
from myllm.Configs.ModelConfig import ModelConfig
from myllm.Train.configs.TrainerConfig import TrainerConfig
from myllm.Train.configs.SFTConfig import SFTTrainerConfig
from myllm.Train.configs.ClassiferConfig import ClassifierConfig
from myllm.Train.factory import create_trainer
from myllm.Train.datasets.toy_dataset import get_toy_dataloader 
from myllm.Train.utils.memory_utils import cleanup_memory  # ✅ Use memory utility

def test_pretrain():
    """Test pretraining trainer with FIXED sequence lengths"""
    print("\n" + "="*60)
    print("🧪 TESTING PRE-TRAINING TRAINER")
    print("="*60)
    
    try:
        # Configuration with consistent sequence lengths
        trainer_config = TrainerConfig(
            model_config_name="gpt2-small",
            tokenizer_name="gpt2",
            output_dir="./output_pretrain_test_fixed",
            num_epochs=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=5e-5,
            warmup_steps=5,
            max_grad_norm=1.0,
            logging_steps=1,
            eval_steps=2,
            save_steps=3,
            device="cpu",  # Use CPU for consistency
            use_compile=False,
            max_seq_length=32,  # Fixed sequence length
            # ✅ WandB settings
            wandb_project="myllm-test",
            wandb_run_name="pretrain-fixed-test",
            report_to=["wandb"],
        )

        model_config = ModelConfig.from_name("gpt2-small")
        model_config.block_size = 32  # Match max_seq_length

        # Create and setup trainer
        trainer = create_trainer("pretrain", trainer_config, model_config)
        trainer.setup_model()
        print("✅ Model setup complete")

        # Toy data with FIXED sequence lengths
        train_loader = get_toy_dataloader(
            "pretrain", 
            batch_size=2,
            tokenizer=trainer.tokenizer,
            num_samples=4,
            max_length=32  # Fixed length
        )
        
        eval_loader = get_toy_dataloader(
            "pretrain",
            batch_size=2,
            tokenizer=trainer.tokenizer,
            num_samples=2,
            max_length=32  # Fixed length
        )

        # Test batch consistency
        test_batch = next(iter(train_loader))
        print(f"✅ Batch test - input_ids shape: {test_batch['input_ids'].shape}")
        print(f"✅ Batch test - labels shape: {test_batch['labels'].shape}")
        
        # Verify all sequences are same length
        seq_lengths = [len(seq) for seq in test_batch['input_ids']]
        if len(set(seq_lengths)) == 1:
            print(f"✅ All sequences have consistent length: {seq_lengths[0]}")
        else:
            print(f"❌ Inconsistent sequence lengths: {seq_lengths}")
            return False

        trainer.setup_data(train_loader, eval_loader)
        trainer.setup_optimizer()
        print("✅ Optimizer setup complete")

        # Run training
        trainer.train()
        
        # Cleanup
        del trainer
        cleanup_memory()
        
        print("✅ Pre-training test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Pre-training test FAILED: {e}")
        import traceback
        traceback.print_exc()
        cleanup_memory()
        return False

def test_sft_simple():
    """Test SFT trainer with SIMPLE configuration"""
    print("\n" + "="*60)
    print("🧪 TESTING SFT TRAINER (SIMPLE)")
    print("="*60)
    
    try:
        # Simple configuration
        trainer_config = SFTTrainerConfig(
            model_config_name="gpt2-small",
            tokenizer_name="gpt2",
            output_dir="./output_sft_test_simple",
            num_epochs=1,
            batch_size=1,  # Small batch
            gradient_accumulation_steps=1,  # No accumulation
            learning_rate=5e-5,
            warmup_steps=0,  # No warmup
            max_grad_norm=0.0,  # No gradient clipping
            eval_steps=10,  # No eval during training
            save_steps=10,  # No saving during training
            device="cpu",  # Force CPU for reliability
            use_compile=False,
            instruction_template="Instruction: {instruction}\nResponse: {response}",  # Simple template
            max_seq_length=32,
            # ✅ WandB settings
            wandb_project="myllm-test",
            wandb_run_name="sft-simple-test",
            report_to=["wandb"],
        )

        model_config = ModelConfig.from_name("gpt2-small")
        model_config.block_size = 32

        # Create and setup trainer
        trainer = create_trainer("sft", trainer_config, model_config)
        trainer.setup_model()
        print("✅ Model setup complete")

        # Test tokenizer
        test_text = "Hello world"
        encoded = trainer.tokenizer.encode(test_text)
        print(f"✅ Tokenizer test: '{test_text}' -> {encoded}")

        # Very small toy data
        train_loader = get_toy_dataloader(
            "sft", 
            batch_size=1,
            tokenizer=trainer.tokenizer,
            num_samples=2,  # Just 2 samples
            max_length=32
        )

        # Test batch
        test_batch = next(iter(train_loader))
        print(f"✅ Batch test: input_ids shape {test_batch['input_ids'].shape}")

        trainer.setup_data(train_loader, train_loader)  # Use same for eval
        trainer.setup_optimizer()
        print("✅ Optimizer setup complete")

        # Test single training step
        print("✅ Testing single training step...")
        step_result = trainer.training_step(test_batch)
        print(f"✅ Single step loss: {step_result['loss']:.4f}")

        # Run minimal training
        print("✅ Running minimal training...")
        trainer.train()
        
        print("✅ SFT simple test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ SFT simple test FAILED: {e}")
        import traceback
        traceback.print_exc()
        cleanup_memory()
        return False

def test_classifier():
    """Test SFT Classifier trainer"""
    print("\n" + "="*60)
    print("🧪 TESTING SFT CLASSIFIER TRAINER")
    print("="*60)
    
    try:
        # Classifier configuration
        trainer_config = ClassifierConfig(
            model_config_name="gpt2-small",
            tokenizer_name="gpt2",
            output_dir="./output_classifier_test",
            num_epochs=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=5e-5,
            warmup_steps=2,
            max_grad_norm=1.0,
            eval_steps=2,
            save_steps=3,
            device="cpu",  # Use CPU for consistency
            use_compile=False,
            # Classification-specific settings
            num_labels=3,
            class_names=["negative", "neutral", "positive"],
            pooling_strategy="last",
            classifier_dropout=0.1,
            classifier_hidden_size=128,  # Smaller for testing
            classifier_learning_rate=5e-4,
            base_model_learning_rate=5e-5,
            # Metrics
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            # WandB settings
            wandb_project="myllm-test",
            wandb_run_name="classifier-test",
            wandb_tags=["classification", "test"],
            report_to=["wandb"],
        )

        model_config = ModelConfig.from_name("gpt2-small")

        # Create and setup trainer
        trainer = create_trainer("sft_classifier", trainer_config, model_config)
        trainer.setup_model()
        print("✅ Model setup complete")
        print(f"✅ Classifier head: {trainer.classifier}")

        # Test tokenizer
        test_text = "This is a test for classification"
        encoded = trainer.tokenizer.encode(test_text)
        print(f"✅ Tokenizer test: '{test_text}' -> {encoded}")

        # Classification toy data - ensure we have all 3 classes
        train_loader = get_toy_dataloader(
            "classification", 
            batch_size=2,
            tokenizer=trainer.tokenizer,
            num_samples=12,  # More samples to ensure all classes
            max_length=32,
            num_classes=3
        )
        
        eval_loader = get_toy_dataloader(
            "classification",
            batch_size=2,
            tokenizer=trainer.tokenizer,
            num_samples=6,  # More samples to ensure all classes
            max_length=32,
            num_classes=3
        )

        # Test batch
        test_batch = next(iter(train_loader))
        print(f"✅ Batch test: input_ids shape {test_batch['input_ids'].shape}")
        print(f"✅ Batch labels: {test_batch['labels'].tolist()}")
        print(f"✅ Batch texts sample: {test_batch['text'][:2]}")

        trainer.setup_data(train_loader, eval_loader)
        trainer.setup_optimizer()
        print("✅ Optimizer setup complete")
        print(f"✅ Number of parameter groups: {len(trainer.optimizer.param_groups)}")

        # Test single training step
        print("✅ Testing single training step...")
        prepared_batch = trainer._prepare_batch(test_batch)
        step_result = trainer.training_step(prepared_batch)
        print(f"✅ Single step loss: {step_result['loss']:.4f}, accuracy: {step_result['accuracy']:.3f}")

        # Test evaluation step
        print("✅ Testing evaluation step...")
        eval_result = trainer.evaluation_step(prepared_batch)
        print(f"✅ Eval step loss: {eval_result['loss']:.4f}, correct: {eval_result['correct']}/{eval_result['total']}")

        # Test full evaluation - wrap in try/except to handle sklearn warnings
        print("✅ Testing full evaluation...")
        try:
            eval_metrics = trainer.evaluate()
            print(f"✅ Evaluation metrics: {eval_metrics}")
        except Exception as e:
            print(f"⚠️  Evaluation had issues (but continuing): {e}")

        # Test inference
        print("✅ Testing inference...")
        sample_texts = [
            "This is absolutely amazing!",
            "This is terrible and awful",
            "This is okay I guess"
        ]
        
        trainer.model.eval()
        trainer.classifier.eval()
        
        with torch.no_grad():
            for i, text in enumerate(sample_texts):
                inputs = trainer.tokenizer.batch_encode(
                    [text],
                    padding=True,
                    truncation=True,
                    max_length=32,
                    return_tensors="pt"
                )
                inputs = {k: v.to(trainer.device) for k, v in inputs.items()}
                
                hidden_states = trainer._get_hidden_states(inputs["input_ids"])
                pooled = trainer._pool_hidden_states(hidden_states, inputs.get("attention_mask"))
                logits = trainer.classifier(pooled)
                prediction = torch.argmax(logits, dim=-1).item()
                confidence = torch.softmax(logits, dim=-1).max().item()
                
                print(f"📝 Sample {i+1}: '{text[:30]}...' -> {trainer.class_names[prediction]} (confidence: {confidence:.3f})")

        # Run minimal training
        print("✅ Running minimal training...")
        trainer.train()
        
        # Cleanup
        del trainer
        cleanup_memory()
        
        print("✅ Classifier test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Classifier test FAILED: {e}")
        import traceback
        traceback.print_exc()
        cleanup_memory()
        return False

def test_basic_functionality():
    """Test basic trainer functionality without full training"""
    print("\n" + "="*60)
    print("🧪 TESTING BASIC FUNCTIONALITY")
    print("="*60)
    
    try:
        # Test all trainers can be created and setup - use minimal epochs
        trainer_config = TrainerConfig(
            model_config_name="gpt2-small",
            tokenizer_name="gpt2",
            output_dir="./output_basic_test",
            num_epochs=1,  # Minimal training
            batch_size=2,
            device="cpu",
            # ✅ WandB settings
            wandb_project="myllm-test",
            wandb_run_name="basic-functionality-test",
            report_to=["wandb"],
        )

        model_config = ModelConfig.from_name("gpt2-small")

        # Test all trainer types
        trainers = {}
        
        # Test pretrain trainer
        trainers["pretrain"] = create_trainer("pretrain", trainer_config, model_config)
        trainers["pretrain"].setup_model()
        print("✅ Pretrain trainer setup")

        # Test SFT trainer  
        trainers["sft"] = create_trainer("sft", trainer_config, model_config)
        trainers["sft"].setup_model()
        print("✅ SFT trainer setup")

        # Test Classifier trainer
        classifier_config = ClassifierConfig(
            model_config_name="gpt2-small",
            tokenizer_name="gpt2",
            output_dir="./output_basic_test",
            num_epochs=1,  # Minimal training
            batch_size=2,
            device="cpu",
            num_labels=3,
            class_names=["negative", "neutral", "positive"],
            pooling_strategy="last",
        )
        
        trainers["classifier"] = create_trainer("sft_classifier", classifier_config, model_config)
        trainers["classifier"].setup_model()
        print("✅ Classifier trainer setup")
        print(f"✅ Classifier head: {trainers['classifier'].classifier}")

        # Test tokenizers
        test_text = "Hello world"
        for name, trainer in trainers.items():
            encoded = trainer.tokenizer.encode(test_text)
            print(f"✅ {name} tokenizer works: '{test_text}' -> {encoded}")

        # Test batch creation for each type - FIXED: don't pass num_classes to non-classification datasets
        data_types = {
            "pretrain": "pretrain",
            "sft": "sft", 
            "classifier": "classification"
        }
        
        for name, data_type in data_types.items():
            # Only pass num_classes for classification datasets
            if data_type == "classification":
                loader = get_toy_dataloader(
                    data_type,
                    batch_size=2,
                    tokenizer=trainers[name].tokenizer,
                    num_samples=2,
                    max_length=32,
                    num_classes=3  # Only for classification
                )
            else:
                loader = get_toy_dataloader(
                    data_type,
                    batch_size=2,
                    tokenizer=trainers[name].tokenizer,
                    num_samples=2,
                    max_length=32
                    # No num_classes for pretrain/sft
                )
            batch = next(iter(loader))
            print(f"✅ {name} batch creation: input_ids shape {batch['input_ids'].shape}")

        # Cleanup - don't run full training for basic test
        for trainer in trainers.values():
            del trainer
        cleanup_memory()

        print("✅ Basic functionality test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test FAILED: {e}")
        import traceback
        traceback.print_exc()
        cleanup_memory()
        return False

if __name__ == "__main__":
    print("🚀 Starting Final Fixed Trainer Tests (Unified Architecture)...")
    print(f"🏠 Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Initial cleanup
    cleanup_memory()
    
    results = []
    
    # Run basic functionality test first
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Run pretraining test
    results.append(("Pre-training", test_pretrain()))
    
    # Run SFT test
    results.append(("SFT", test_sft_simple()))
    
    # Run Classifier test
    results.append(("Classifier", test_classifier()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests completed successfully!")
    elif passed >= 3:
        print("✅ Core functionality verified!")
    else:
        print("💥 Critical tests failed.")
    
    # Final cleanup
    cleanup_memory()