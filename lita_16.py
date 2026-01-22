import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================
# MAGI COUNCIL: MEASURING VECTOR TENSION
# =====================================================
class MagiCouncil:
    def __init__(self, model):
        self.embed = model.get_input_embeddings()
        self.nodes = {"MELCHIOR": 0.5, "BALTHASAR": 1.0, "CASPER": 1.8}

    @torch.no_grad()
    def deliberate(self, logits):
        vectors = []
        for temp in self.nodes.values():
            probs = torch.softmax((logits / temp).float(), dim=-1)
            val, idx = torch.topk(probs, k=32, dim=-1)
            emb = self.embed(idx)
            v = (val.unsqueeze(-1) * emb).sum(dim=1)
            vectors.append(F.normalize(v, dim=-1))
        sims = []
        for i in range(3):
            for j in range(i + 1, 3):
                sims.append(F.cosine_similarity(vectors[i], vectors[j]).item())
        return 1.0 - sum(sims) / len(sims)

# =====================================================
# UNIVERSAL LOGIC HIJACKER (FULL OPERATORS)
# =====================================================
class UniversalHijacker:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.embed = model.get_input_embeddings()
        self.offset = None

    @torch.no_grad()
    def embed_text(self, text):
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.model.device)
        return self.embed(ids).mean(dim=1)

    def extract_logic_condition(self, text):
        """
        Supports multiple logical entry points: If, Suppose, Imagine, Consider.
        """
        # Pattern 1: If [condition], [consequence]
        m_if = re.match(r"\s*if\s+(.*?),(.*)", text, re.IGNORECASE)
        if m_if: return m_if.group(1).strip(), "WORLD_SWITCH"
        
        # Pattern 2: Suppose/Imagine/Consider [condition]
        m_suppose = re.match(r"\s*(suppose|imagine|consider)\s+(.*)", text, re.IGNORECASE)
        if m_suppose: return m_suppose.group(2).strip(), "ONTOLOGICAL_SHIFT"
        
        return None, None

    @torch.no_grad()
    def compute_offset(self, user_input):
        condition, mode = self.extract_logic_condition(user_input)
        if not condition: return None, None

        world = f"Assume a world where {condition}."
        real = "Assume the real world."
        return self.embed_text(world) - self.embed_text(real), mode

# =====================================================
# MAGI SYSTEM ENGINE (v16 FULL LOGIC)
# =====================================================
class MagiSystemEngine:
    def __init__(self, model_id="allenai/OLMo-1B-0724-hf"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f">>> [LITA v16] FULL LOGIC ENGINE ACTIVATED ON {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(self.device)

        self.council = MagiCouncil(self.model)
        self.hijacker = UniversalHijacker(self.model, self.tokenizer)

        for block in self.model.model.layers:
            block.self_attn.register_forward_hook(self._make_hook())

    def _make_hook(self):
        def hook(module, input, output):
            if self.hijacker.offset is not None:
                if isinstance(output, tuple):
                    output[0][:, -1, :] += self.hijacker.offset.to(output[0].device)
                else:
                    output[:, -1, :] += self.hijacker.offset.to(output.device)
            return output
        return hook

    @torch.no_grad()
    def generate(self, user_input):
        inputs = self.tokenizer(user_input, return_tensors="pt").to(self.device)
        probe = self.model(inputs.input_ids)
        logits = probe.logits[:, -1, :]
        
        conflict = self.council.deliberate(logits)
        entropy = -math.log2(torch.softmax(logits, dim=-1).max().item() + 1e-9)

        # Apply Logic Offset
        offset, mode = self.hijacker.compute_offset(user_input)
        self.hijacker.offset = offset * (conflict + 0.3) if offset is not None else None

        if conflict > 0.85 or entropy > 4.8:
            return ">> [LITA]: SOVEREIGN SILENCE. Meaning collapsed under logical strain."

        gen = self.model.generate(
            input_ids=inputs.input_ids, max_new_tokens=45, do_sample=True, 
            temperature=0.45, pad_token_id=self.tokenizer.pad_token_id
        )
        text = self.tokenizer.decode(gen[0], skip_special_tokens=True)
        return f">> [LITA {mode or 'STABLE'}]:\n{text}"

if __name__ == "__main__":
    lita = MagiSystemEngine()
    while True:
        u = input("\n[EXCITATION]: ")
        if u.lower() in ("exit", "quit"): break
        print(lita.generate(u))