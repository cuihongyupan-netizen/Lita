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
    Three-node Sovereign Council:
    Measures vector tension as a proxy for logical conflict.
    """
    def __init__(self, model):
        # Reference to the model's input embeddings
        self.embed = model.get_input_embeddings()
        # Define council nodes with different temperatures
        self.nodes = {
            "MELCHIOR": 0.4,    # Rational / Conservative
            "BALTHASAR": 0.9,   # Consensus / Social
            "CASPER": 1.6       # Radical / Divergent
        }

    @torch.no_grad()
    def deliberate(self, logits):
        """
        Compute conflict score based on top-k embedding similarities across nodes.
        """
        vectors = []

        for _, temp in self.nodes.items():
            # Apply node temperature to logits
            probs = torch.softmax((logits / temp).float(), dim=-1)
            # Take top-32 probable tokens
            val, idx = torch.topk(probs, k=32, dim=-1)
            # Convert token indices to embeddings
            emb = self.embed(idx)               # (1, k, d)
            # Weighted sum of embeddings
            v = (val.unsqueeze(-1) * emb).sum(dim=1)  # (1, d)
            # Normalize the vector
            vectors.append(F.normalize(v, dim=-1))

        # Compute pairwise cosine similarities
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
    Maps counterfactual or informal logic into embedding offsets
    that can be applied via hooks to the model.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.embed = model.get_input_embeddings()
        self.offset = None  # Stores embedding displacement for hook

    @torch.no_grad()
    def embed_text(self, text):
        """
        Convert text to mean embedding.
        """
        ids = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(self.model.device)
        return self.embed(ids).mean(dim=1)

    @torch.no_grad()
    def compute_offset(self, user_input):
        """
        Detect counterfactual patterns and compute embedding offsets.
        """
        t = user_input.lower()

        # Location counterfactual
        if "were in" in t:
            return self.contextual_offset(
                user_input,
                "Assume the location is different from reality."
            ), "LOC_COUNTERFACTUAL"

        # Property inversion
        if "were cold" in t:
            return self.contextual_offset(
                user_input,
                "Assume thermal properties are inverted."
            ), "PROP_INVERSION"

        # Category shift (ontological)
        if "as a" in t:
            return self.contextual_offset(
                user_input,
                "Assume a different ontological category."
            ), "CATEGORY_SHIFT"

        return None, None

    @torch.no_grad()
    def contextual_offset(self, base_prompt, counterfactual_clause):
        """
        Compute embedding difference between counterfactual and base prompt:
        Δ = E(base + counterfactual) - E(base)
        """
        base = self.embed_text(base_prompt)
        altered = self.embed_text(base_prompt + " " + counterfactual_clause)
        return altered - base

# =====================================================
# MAGI SYSTEM ENGINE WITH HOOK
# =====================================================
class MagiSystemEngine:
    """
    The main engine combining the model, council, and hijacker.
    Applies hooks dynamically for counterfactual logic manipulation.
    """
    def __init__(self, model_id="allenai/OLMo-1B-0724-hf"):
        # Choose device (MPS for Apple Silicon if available)
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f">>> [INIT] LITA UNIVERSAL ENGINE ACTIVATED ON {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float32,  # Recommended for MPS stability
        ).to(self.device)

        # Initialize hijacker and council
        self.hijacker = UniversalHijacker(self.model, self.tokenizer)
        self.council = MagiCouncil(self.model)

        # ---------- ATTACH HOOKS TO ATTENTION LAYERS ----------
        self.hooks = []
        for i, block in enumerate(self.model.model.layers):
            # OLMo attention layer is usually called self_attn
            hook = block.self_attn.register_forward_hook(self._make_hook())
            self.hooks.append(hook)

    def _make_hook(self):
        """
        Returns a forward hook function that adds offset to the last token's embedding.
        """
        def hook(module, input, output):
            if self.hijacker.offset is not None:
                # Handle tuple outputs (typical for OLMo)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    hidden_states[:, -1, :] = hidden_states[:, -1, :] + self.hijacker.offset.to(hidden_states.device)
                    return (hidden_states,) + output[1:]
                else:
                    output[:, -1, :] = output[:, -1, :] + self.hijacker.offset.to(output.device)
            return output
        return hook

    @torch.no_grad()
    def generate(self, user_input):
        """
        Generate text from user input, applying hijacker hooks if conditions are met.
        """
        # Tokenize input
        inputs = self.tokenizer(
            user_input,
            return_tensors="pt"
        ).to(self.device)

        # ---------- PROBING ----------
        out = self.model(inputs.input_ids)
        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)

        # Calculate peak entropy for Sovereign Silence mechanism
        peak_entropy = -math.log2(probs.max().item() + 1e-9)
        # Compute internal council conflict
        conflict = self.council.deliberate(logits)

        # ---------- LOGICAL HIJACKING ----------
        offset, logic_type = self.hijacker.compute_offset(user_input)
        self.hijacker.offset = None  # Reset offset
        if offset is not None:
            # Scale hijacking strength based on conflict
            strength = min(1.0, conflict * 0.8)
            self.hijacker.offset = offset * strength
            print(f"[HIJACK] {logic_type} confirmed -> applying dynamic hook intervention")

        # ---------- GENERATION ----------
        gen = self.model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.45,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode output tokens
        text = self.tokenizer.decode(gen[0], skip_special_tokens=True)

        # ---------- DEBUG OUTPUT ----------
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
            # Clear MPS cache if on Apple Silicon
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
