import torch
import yaml
from pathlib import Path

def test_imports():
    print("Testing imports...")
    try:
        from transformers import Trainer, TrainingArguments
        from dl_solution import CustomTrainer, compute_metrics_fn, TransformerClassifier
        print("✅ Basic imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        from dl_solution_advanced import CustomTrainer as CustomTrainerAdv, FGM
        print("✅ Advanced imports successful")
    except Exception as e:
        print(f"❌ Advanced import error: {e}")
        return False
    
    return True

def test_config_loading():
    print("\nTesting config loading...")
    try:
        config_path = Path('./configs/dl_graphcodebert_deepspeed.yaml')
        if not config_path.exists():
            print("⚠️  DeepSpeed config not found, using default")
            config_path = Path('./configs/dl_graphcodebert.yaml')
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Config loaded from {config_path}")
        print(f"   Model: {config.get('model_name', 'N/A')}")
        print(f"   DeepSpeed: {config.get('deepspeed', 'Not configured')}")
        return True
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False

def test_deepspeed_configs():
    print("\nTesting DeepSpeed configurations...")
    configs = [
        './configs/ds_config_zero1.json',
        './configs/ds_config_zero2.json',
        './configs/ds_config_zero3.json'
    ]
    
    all_exist = True
    for config_path in configs:
        if Path(config_path).exists():
            print(f"✅ {config_path} exists")
        else:
            print(f"❌ {config_path} not found")
            all_exist = False
    
    return all_exist

def test_model_instantiation():
    print("\nTesting model instantiation...")
    try:
        from dl_solution import TransformerClassifier
        
        model = TransformerClassifier(
            model_name="microsoft/codebert-base",
            num_labels=16,
            dropout=0.1,
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05
        )
        
        print("✅ Model instantiated successfully")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Num labels: {model.num_labels}")
        
        dummy_input = {
            'input_ids': torch.randint(0, 1000, (2, 128)),
            'attention_mask': torch.ones(2, 128)
        }
        
        output = model(**dummy_input)
        print(f"✅ Forward pass successful, output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ Model instantiation error: {e}")
        return False

def test_trainer_setup():
    print("\nTesting Trainer setup...")
    try:
        from transformers import TrainingArguments
        from dl_solution import CustomTrainer, CodeCommentDataset, AsymmetricLoss
        from transformers import AutoTokenizer
        import numpy as np
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        
        dummy_texts = ["def hello(): pass", "print('world')"]
        dummy_labels = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        
        dataset = CodeCommentDataset(dummy_texts, dummy_labels, tokenizer, max_len=128)
        
        from dl_solution import TransformerClassifier
        model = TransformerClassifier(
            model_name="microsoft/codebert-base",
            num_labels=16,
            use_lora=False
        )
        
        training_args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            logging_steps=1,
            report_to=[]
        )
        
        criterion = AsymmetricLoss()
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            custom_loss_fn=criterion
        )
        
        print("✅ Trainer setup successful")
        print(f"   Trainer type: {type(trainer).__name__}")
        return True
    except Exception as e:
        print(f"❌ Trainer setup error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("Testing Updated Training Scripts")
    print("="*60)
    
    results = {
        "Imports": test_imports(),
        "Config Loading": test_config_loading(),
        "DeepSpeed Configs": test_deepspeed_configs(),
        "Model Instantiation": test_model_instantiation(),
        "Trainer Setup": test_trainer_setup()
    }
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to use Trainer API with DeepSpeed.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    import shutil
    if Path("./test_output").exists():
        shutil.rmtree("./test_output")
        print("\n🧹 Cleaned up test output directory")

if __name__ == "__main__":
    main()

