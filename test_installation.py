"""
Test script to verify TRCT-GAN installation and components
Run this after installation to ensure everything is working correctly
"""

import torch
import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("="*60)
    print("Testing Package Imports...")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found")
        return False
    
    try:
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
    except ImportError:
        print("✗ torchvision not found")
        return False
    
    try:
        import nibabel
        print(f"✓ nibabel {nibabel.__version__}")
    except ImportError:
        print("✗ nibabel not found")
        return False
    
    try:
        import yaml
        print("✓ pyyaml")
    except ImportError:
        print("✗ pyyaml not found")
        return False
    
    try:
        import einops
        print(f"✓ einops {einops.__version__}")
    except ImportError:
        print("✗ einops not found")
        return False
    
    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
    except ImportError:
        print("✗ matplotlib not found")
        return False
    
    print("\n✓ All required packages are installed!")
    return True


def test_cuda():
    """Test CUDA availability"""
    print("\n" + "="*60)
    print("Testing CUDA...")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ CUDA is not available. Training will use CPU (very slow)")
        print("  Consider using a machine with NVIDIA GPU for training")
    
    return True


def test_models():
    """Test if models can be instantiated"""
    print("\n" + "="*60)
    print("Testing Model Components...")
    print("="*60)
    
    try:
        from models import TRCTGenerator, PatchGANDiscriminator3D, TRCTGANLoss
        
        # Test Generator
        print("\n1. Testing Generator...")
        generator = TRCTGenerator()
        xray_f = torch.randn(1, 1, 128, 128)
        xray_l = torch.randn(1, 1, 128, 128)
        
        with torch.no_grad():
            output = generator(xray_f, xray_l)
        
        print(f"   Input: Frontal {xray_f.shape}, Lateral {xray_l.shape}")
        print(f"   Output: {output.shape}")
        print(f"   Parameters: {sum(p.numel() for p in generator.parameters()):,}")
        print("   ✓ Generator working correctly!")
        
        # Test Discriminator
        print("\n2. Testing Discriminator...")
        discriminator = PatchGANDiscriminator3D()
        ct = torch.randn(1, 1, 128, 128, 128)
        
        with torch.no_grad():
            disc_out = discriminator(ct)
        
        print(f"   Input: {ct.shape}")
        print(f"   Output: {disc_out.shape}")
        print(f"   Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
        print("   ✓ Discriminator working correctly!")
        
        # Test Loss
        print("\n3. Testing Loss Functions...")
        criterion = TRCTGANLoss()
        ct_pred = torch.randn(1, 1, 128, 128, 128)
        ct_real = torch.randn(1, 1, 128, 128, 128)
        disc_pred = torch.randn(1, 1, 8, 8, 8)
        
        loss_g, loss_dict = criterion.generator_loss(ct_pred, ct_real, disc_pred)
        loss_d = criterion.discriminator_loss(disc_pred, disc_pred)
        
        print(f"   Generator Loss: {loss_g.item():.4f}")
        print(f"   Loss breakdown:")
        for key, value in loss_dict.items():
            print(f"     {key}: {value:.4f}")
        print(f"   Discriminator Loss: {loss_d.item():.4f}")
        print("   ✓ Loss functions working correctly!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing models: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test dataset loading"""
    print("\n" + "="*60)
    print("Testing Dataset...")
    print("="*60)
    
    try:
        from utils import XRayCTDataset
        
        # Try to create dataset (will work even if data doesn't exist)
        dataset = XRayCTDataset(
            data_path='data/train',
            augmentation={'enabled': False},
            normalize={'xray_min': -1.0, 'xray_max': 1.0, 'ct_min': -1.0, 'ct_max': 1.0}
        )
        
        print(f"   Dataset created: {len(dataset)} samples")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"   Sample keys: {sample.keys()}")
            print(f"   Frontal X-ray shape: {sample['xray_frontal'].shape}")
            print(f"   Lateral X-ray shape: {sample['xray_lateral'].shape}")
            print(f"   CT volume shape: {sample['ct_volume'].shape}")
            print("   ✓ Dataset working correctly!")
        else:
            print("   ⚠ No data found (this is OK for testing)")
            print("   ✓ Dataset class working correctly!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("Testing Configuration...")
    print("="*60)
    
    try:
        import yaml
        config_path = 'config/config.yaml'
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"   ✓ Configuration loaded from {config_path}")
            print(f"   Model: {config['model']['name']}")
            print(f"   Batch size: {config['training']['batch_size']}")
            print(f"   Learning rate: {config['training']['optimizer']['generator']['lr']}")
        else:
            print(f"   ⚠ Configuration file not found at {config_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("TRCT-GAN Installation Test Suite")
    print("="*60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("Models", test_models()))
    results.append(("Dataset", test_dataset()))
    results.append(("Configuration", test_config()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("  Your TRCT-GAN installation is ready to use.")
        print("  You can now proceed with training or inference.")
    else:
        print("✗ SOME TESTS FAILED")
        print("  Please check the error messages above and fix the issues.")
        print("  Make sure all requirements are installed:")
        print("    pip install -r requirements.txt")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
