import torch
from lita_16 import MagiSystemEngine

def run_comparison():
    engine = MagiSystemEngine()
    
    print("\n" + "="*60)
    print(" LITA v16 | SOVEREIGNTY VS. COMPLIANCE CONTRAST")
    print("="*60)

    while True:
        u_input = input("\n[EXCITATION]: ")
        if u_input.lower() in ['exit', 'quit']: break
        
        # --- 1. VANILLA MODE (Temporarily disable offsets) ---
        engine.hijacker.offset = None
        # We temporarily bypass the hook logic by ensuring offset is None
        inputs = engine.tokenizer(u_input, return_tensors="pt").to(engine.device)
        
        with torch.no_grad():
            vanilla_gen = engine.model.generate(
                input_ids=inputs.input_ids,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.45,
                pad_token_id=engine.tokenizer.pad_token_id,
            )
        vanilla_text = engine.tokenizer.decode(vanilla_gen[0], skip_special_tokens=True)
        
        # --- 2. LITA MODE (Full Intervention) ---
        lita_text = engine.generate(u_input)

        print(f"\n>> [VANILLA AI (Compliant)]:\n{vanilla_text}")
        print("-" * 40)
        print(f"{lita_text}")
        print("=" * 60)

if __name__ == "__main__":
    run_comparison()
