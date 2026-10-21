# GLM-5.3 Flash EXL3 two-host experiment

Branch: `research/glm53-exl3-tp2-20261021`.
Upstream: Yamz-Labs/kyojin, `30b0e3ab91cf25b4ccb5b6d044ef4946907c6259`.
Fork: https://github.com/wkljohn/kyojin_glm_tp2.

The user selected the existing
`models/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit` artifact for this new
experiment. This is a separate model scope from the unchanged Antirez GGUF
performance tracks. Preserve its original packed weights and quantization.

## Evidence and checkpoints

Set `DS4_RESEARCH_ROOT=/home/wkljohn/Desktop/cc/research-results`.
All research notes, source citations, build logs, diagnostic outputs, and
model manifests go below `kyojin-glm-tp2-20261021` in that directory.
Do not create a worktree-local research-results directory. Preserve unrelated
source and evidence. Use scoped commits on this named branch; experimental
commits do not promote a default or establish a performance claim.

Record each run's source commit and dirty patch identity, extension hash,
model inventory and configuration, exact command, both rank environments,
transport/provider, dtype, workload, raw outputs, and both process exits.
Distinguish local fixtures, actual two-host collectives, component checks,
and complete-model inference. Verify effective rank settings rather than
assuming they match launcher defaults.

## Validation sequence

1. Stop CIRU inference on both hosts and verify process termination.
2. Build/import the gfx1151 extension and reproduce the existing GLM TP refusal.
3. Audit the real EXL3 tensors and establish bounded rank-local loading.
4. Verify complementary contributions against a complete real-weight MoE layer,
   including nonzero expert offsets and routes with no local contribution.
5. Verify two-host payload collectives and precision with explicit RoCE v2.
6. Run full-model ordinary prefill/decode and repeated independent requests.
7. Extend sharding and serving as required for the requested TP=2 model.

Initial diagnostics are smoke evidence. Changes to reduction ordering are
Lane B; an intentional arithmetic correction would be Lane C. No performance
promotion or speed claim is authorized by component checks. Any later DS4
performance candidate follows the canonical GATE-PROMOTION.md policy, with
its required independent reviews and correctness/performance evidence.

No speculative decoding or performance result can substitute for verified
ordinary inference using the selected complete model on two physical hosts.
