# examples/run_all_tests_fixed_final.py (REFACTORED)
"""
Final fixed test script with proper error handling using unified architecture
"""

import torch
import gc
from myllm.Configs.ModelConfig import ModelConfig
from myllm.Train.configs.TrainerConfig import TrainerConfig
from myllm.Train.configs.SFTConfig import SFTTrainerConfig
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

def test_basic_functionality():
    """Test basic trainer functionality without full training"""
    print("\n" + "="*60)
    print("🧪 TESTING BASIC FUNCTIONALITY")
    print("="*60)
    
    try:
        # Test both trainers can be created and setup
        trainer_config = TrainerConfig(
            model_config_name="gpt2-small",
            tokenizer_name="gpt2",
            output_dir="./output_basic_test",
            num_epochs=0,  # No training
            batch_size=2,
            device="cpu",
            # ✅ WandB settings (but won't initialize due to num_epochs=0)
            wandb_project="myllm-test",
            wandb_run_name="basic-functionality-test",
            report_to=["wandb"],
        )

        model_config = ModelConfig.from_name("gpt2-small")

        # Test pretrain trainer
        pretrain_trainer = create_trainer("pretrain", trainer_config, model_config)
        pretrain_trainer.setup_model()
        print("✅ Pretrain trainer setup")

        # Test SFT trainer  
        sft_trainer = create_trainer("sft", trainer_config, model_config)
        sft_trainer.setup_model()
        print("✅ SFT trainer setup")

        # Test tokenizers
        test_text = "Hello world"
        pretrain_encoded = pretrain_trainer.tokenizer.encode(test_text)
        sft_encoded = sft_trainer.tokenizer.encode(test_text)
        print(f"✅ Tokenizers work: '{test_text}' -> {pretrain_encoded}")

        # Test batch creation
        train_loader = get_toy_dataloader(
            "pretrain", 
            batch_size=2,
            tokenizer=pretrain_trainer.tokenizer,
            num_samples=2
        )
        test_batch = next(iter(train_loader))
        print(f"✅ Batch creation: input_ids shape {test_batch['input_ids'].shape}")

        # Test SFT-specific functionality
        sft_train_loader = get_toy_dataloader(
            "sft",
            batch_size=2,
            tokenizer=sft_trainer.tokenizer,
            num_samples=2
        )
        sft_test_batch = next(iter(sft_train_loader))
        print(f"✅ SFT batch creation: input_ids shape {sft_test_batch['input_ids'].shape}")

        # Cleanup
        del pretrain_trainer, sft_trainer
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
    elif passed >= 2:
        print("✅ Core functionality verified!")
    else:
        print("💥 Critical tests failed.")
    
    # Final cleanup
    cleanup_memory()