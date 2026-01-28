import torch
import numpy as np
from src.models import get_model
from src.engine import Ranger, quantize_model, save_model
import os

def test_everything():
    print("🧪 STARTING REPO INTEGRITY TEST...")
    
    # 1. Test Architecture Instantiation
    print("   1. Testing Model Creation...", end="")
    try:
        model = get_model("winner", input_size=32, hidden_size=240, layers=6)
        print("PASS ✅")
    except Exception as e:
        print(f"FAIL ❌\n{e}")
        return

# 2. Test Forward Pass (FP32)
    print("   2. Testing Forward Pass...", end="")
    try:
        dummy_x = torch.randn(2, 100, 32) # Batch, Time, Feat
        out, _ = model(dummy_x)
        
        # Check if output is (Batch, Time, 32) OR (Batch, 32)
        if len(out.shape) == 3:
            assert out.shape == (2, 100, 32), f"Shape Mismatch: {out.shape}"
            # For loss calculation test, we take the last step or full seq
            out_for_loss = out[:, -1, :] 
        else:
            assert out.shape == (2, 32), f"Shape Mismatch: {out.shape}"
            out_for_loss = out
            
        print("PASS ✅")
    except Exception as e:
        print(f"FAIL ❌\n{e}")
        return

    # 3. Test Optimizer (Ranger)
    print("   3. Testing Training Step...", end="")
    try:
        opt = Ranger(model.parameters())
        crit = torch.nn.MSELoss()
        target = torch.randn(2, 32)
        
        # Use the processed output for loss
        loss = crit(out_for_loss, target) 
        loss.backward()
        opt.step()
        print("PASS ✅")
    except Exception as e:
        print(f"FAIL ❌\n{e}")
        return

    # 4. Test Quantization
    print("   4. Testing INT8 Quantization...", end="")
    try:
        q_model = quantize_model(model)
        # Run inference on quantized
        with torch.no_grad():
            q_out, _ = q_model(dummy_x)
        print("PASS ✅")
    except Exception as e:
        print(f"FAIL ❌\n{e}")
        return
        
    # 5. Test Saving/Loading
    print("   5. Testing Save/Load...", end="")
    try:
        save_model(q_model, "test_q.pt")
        # Load back
        model_new = get_model("winner")
        q_model_new = quantize_model(model_new)
        state = torch.load("test_q.pt", weights_only=False)
        q_model_new.load_state_dict(state)
        os.remove("test_q.pt")
        print("PASS ✅")
    except Exception as e:
        print(f"FAIL ❌\n{e}")
        return

    print("\n🎉 ALL SYSTEMS GO. REPO IS READY FOR DEPLOYMENT.")

if __name__ == "__main__":
    test_everything()