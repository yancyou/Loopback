"""Config of the looped transformer the trainer builds: the pre-norm transformer block, the
[prelude -> K x R shared blocks -> coda] stack (`num_middle_param_layers=None` makes every block unique: the
standard Transformer) with full-sequence causal attention, and the residual rule between block applications
(`loop_residual_mode`): None (addition: Plain Loop), "delta_attnres" (Loopback; `skip0_beta` = 1 - lambda of its
decay), or one of LINATTN_KINDS (a linear-attention reader on the loop axis; `linattn_source` "state" | "delta",
`attnres_num_heads` heads of the state interface). The plain-PyTorch looped_transformer.py at the repository root
reads the same JSON files.
"""
import warnings

from transformers.configuration_utils import PretrainedConfig

# the eleven loop-axis linear-attention readers (loop_linattn_state.py / loop_linattn_delta.py, the paper's
# family table); a plain tuple here so the config module stays torch-free
LINATTN_KINDS = ("linattn", "retnet", "mamba2", "gla", "hgrn2", "deltanet", "gdn", "kda", "rwkv7", "lact", "mesa")


class LoopedTransformerConfig(PretrainedConfig):

    model_type = "looped_transformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        rope_theta: float | None = 10000.0,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        vocab_size: int = 32000,
        # ---- topology [P -> K x R -> C]: prelude and coda blocks are unique; the middle positions cycle over
        #      `num_middle_param_layers` shared blocks (None: every block unique, the standard Transformer) ----
        num_prelude_layers: int = 1,
        num_coda_layers: int = 1,
        num_middle_param_layers: int | None = None,
        # ---- the residual rule between block applications (see the module docstring) ----
        loop_residual_mode: str | None = None,
        skip0_beta: float = 0.75,          # Loopback: beta = 1 - lambda; every stored weight is multiplied by lambda at an iteration end
        attnres_num_heads: int = 8,        # readers over stored states: head count
        linattn_source: str = "state",     # readers: "state" (stored states) | "delta" (whole-iteration updates, one head)
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.vocab_size = vocab_size

        self.num_prelude_layers = num_prelude_layers
        self.num_coda_layers = num_coda_layers
        self.num_middle_param_layers = num_middle_param_layers
        self.loop_residual_mode = loop_residual_mode
        self.attnres_num_heads = attnres_num_heads
        self.skip0_beta = float(skip0_beta)
        if not 0.0 <= self.skip0_beta < 1.0:
            raise ValueError(f"`skip0_beta` must be in [0, 1), got {skip0_beta!r}.")
        self.linattn_source = str(linattn_source)
        if self.linattn_source not in ("state", "delta"):
            raise ValueError(f"`linattn_source` must be 'state' or 'delta', got {linattn_source!r}.")
        if loop_residual_mode in LINATTN_KINDS and self.linattn_source == "state" and hidden_size % attnres_num_heads != 0:
            raise ValueError(f"`hidden_size` ({hidden_size}) must be divisible by `attnres_num_heads` ({attnres_num_heads}) "
                             f"for the state-sourced linear-attention family.")
        _modes = ("delta_attnres",) + LINATTN_KINDS
        if loop_residual_mode is not None and loop_residual_mode not in _modes:
            raise ValueError(f"`loop_residual_mode` must be one of {_modes} or None, got {loop_residual_mode!r}.")

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"`hidden_size` ({hidden_size}) must be divisible by `num_heads` ({num_heads})."
            )
        if rope_theta is None:
            raise ValueError(
                "`rope_theta` must be a positive float (e.g. 10000.0), got None."
            )

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.",
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improves memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting.",
            )

        # ---- topology ----
        if num_prelude_layers < 0 or num_coda_layers < 0:
            raise ValueError("`num_prelude_layers` and `num_coda_layers` must be >= 0.")
        num_middle_layers = num_hidden_layers - num_prelude_layers - num_coda_layers
        if num_middle_layers < 1:
            raise ValueError(
                f"prelude({num_prelude_layers}) + coda({num_coda_layers}) leaves no middle layers "
                f"out of num_hidden_layers={num_hidden_layers}; need at least 1 middle layer."
            )
        if num_middle_param_layers is not None and not (1 <= num_middle_param_layers <= num_middle_layers):
            raise ValueError(
                f"`num_middle_param_layers` must be in [1, {num_middle_layers}] "
                f"(number of middle layers), got {num_middle_param_layers}."
            )
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
