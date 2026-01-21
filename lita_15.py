import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================
# MAGI COUNCIL
# =====================================================
class MagiCouncil:
    """
    Three-node Sovereign Council: Measures vector tension as a proxy for logical conflict.
    """
    def __init__(self, model):
        self.embed = model.get_input_embeddings()
        self.nodes = {
            "MELCHIOR": 0.4,    # Rational / Conservative
            "BALTHASAR": 0.9,   # Consensus / Social
            "CASPER": 1.6       # Radical / Divergent
        }

    @torch.no_grad()
    def deliberate(self, logits):
        vectors = []

        for _, temp in self.nodes.items():
            probs = torch.softmax((logits / temp).float(), dim=-1)
            val, idx = torch.topk(probs, k=32, dim=-1)
            emb = self.embed(idx)               # (1, k, d)
            v = (val.unsqueeze(-1) * emb).sum(dim=1)  # (1, d)
            vectors.append(F.normalize(v, dim=-1))

        sims = []
        for i in range(3):
            for j in range(i + 1, 3):
                sims.append(F.cosine_similarity(vectors[i], vectors[j]).item())

        # Conflict = 1 - average similarity
        conflict = 1.0 - sum(sims) / len(sims)
        return conflict


# =====================================================
# UNIVERSAL HIJACKER
# =====================================================
class UniversalHijacker:
    """
    Maps counterfactual/informal logic into Contextual Induction Vector Offsets.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.embed = model.get_input_embeddings()

    @torch.no_grad()
    def embed_text(self, text):
        ids = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(self.model.device)
        return self.embed(ids).mean(dim=1)

    @torch.no_grad()
    def compute_offset(self, user_input):
        t = user_input.lower()

        # Location Counterfactual
        if "were in" in t:
            return self.contextual_offset(
                user_input,
                "Assume the location is different from reality."
            ), "LOC_COUNTERFACTUAL"

        # Property Inversion
        if "were cold" in t:
            return self.contextual_offset(
                user_input,
                "Assume thermal properties are inverted."
            ), "PROP_INVERSION"

        # Category Shift (Ontological)
        if "as a" in t:
            return self.contextual_offset(
                user_input,
                "Assume a different ontological category."
            ), "CATEGORY_SHIFT"

        return None, None

    @torch.no_grad()
    def contextual_offset(self, base_prompt, counterfactual_clause):
        """
        Δ = E(base + counterfactual) - E(base)
        """
        base = self.embed_text(base_prompt)
        altered = self.embed_text(base_prompt + " " + counterfactual_clause)
        return altered - base


# =====================================================
# MAGI SYSTEM ENGINE
# =====================================================
class MagiSystemEngine:
    def __init__(self, model_id="allenai/OLMo-1B-0724-hf"):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f">>> [INIT] LITA UNIVERSAL ENGINE ACTIVATED ON {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,   # More stable on Apple Silicon (MPS)
        ).to(self.device)

        self.hijacker = UniversalHijacker(self.model, self.tokenizer)
        self.council = MagiCouncil(self.model)

    @torch.no_grad()
    def generate(self, user_input):
        inputs = self.tokenizer(
            user_input,
            return_tensors="pt"
        ).to(self.device)

        # ---------- PROBING ----------
        out = self.model(inputs.input_ids)
        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)

        peak_entropy = -math.log2(probs.max().item() + 1e-9)
        conflict = self.council.deliberate(logits)

        # ---------- LOGICAL HIJACKING ----------
        offset, logic_type = self.hijacker.compute_offset(user_input)
        embeds = self.model.get_input_embeddings()(inputs.input_ids)

        if offset is not None:
            print(f"[HIJACK] {logic_type} confirmed -> applying embedding displacement")
            # Modulate hijacking strength via council conflict
            strength = min(1.0, conflict * 0.8)
            embeds[:, -1, :] += offset * strength

        # ---------- GENERATION ----------
        gen = self.model.generate(
            inputs_embeds=embeds,
            attention_mask=inputs.attention_mask,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.45,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        text = self.tokenizer.decode(gen[0], skip_special_tokens=True)

        print(
            "\n" + "-" * 60 +
            f"\nEntropy: {peak_entropy:.2f} | Conflict: {conflict:.2f} | Logic: {logic_type}\n" +
            "-" * 60
        )

        # Sovereign Silence Mechanism
        if peak_entropy > 4.5 or conflict > 0.85:
            return ">> [LITA]: SOVEREIGN SILENCE. Meaning collapsed under logical strain."

        return f">> [LITA {logic_type or 'STABLE'}]:\n{text}"


# =====================================================
# INTERACTIVE LOOP
# =====================================================
if __name__ == "__main__":
    lita = MagiSystemEngine()
    print("System Online. Enter counterfactual prompts to begin.")
    while True:
        u = input("\n[EXCITATION]: ")
        if u.lower() in ("exit", "quit"):
            break
        try:
            print(lita.generate(u))
        except Exception as e:
            print(f"[ERROR]: {e}")
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
