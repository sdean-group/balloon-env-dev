# Synchronized Diffusion Alternatives

These experiments keep the conditional wind denoiser, EDM schedule, canonical chart
geometry, coordinate noise, conditions, and query protocol fixed. Only the rule that
reconciles overlapping local diffusion paths changes.

## SyncTweedies adaptation

Each factor retains a local noisy state. At every solver evaluation, factors predict
clean fields, the clean predictions are mapped to chart coordinates and averaged, and
the consensus clean field defines each factor's next EDM direction.

Reference: Kim et al., *SyncTweedies: A General Generative Framework Based on
Synchronized Diffusions*, 2024, https://arxiv.org/abs/2403.14370.

## Overlap-guided adaptation

Each factor follows its own EDM direction. A differentiable overlap objective measures
disagreement between predicted clean fields. Its gradient with respect to each noisy
factor state is RMS-normalized and added as a descent guidance term. This uses raw wind
overlap disagreement instead of SyncDiffusion's image-perceptual loss.

Reference: Lee et al., *SyncDiffusion: Coherent Montage via Synchronized Joint
Diffusions*, NeurIPS 2023,
https://papers.neurips.cc/paper_files/paper/2023/hash/9ee3a664ccfeabc0da16ac6f1f1cfe59-Abstract-Conference.html.

## Fixed-round consensus-equilibrium adaptation

At every solver evaluation, local denoising alternates with a global weighted consensus
field and local dual disagreement corrections. The implementation uses a fixed number
of rounds so random-access work remains bounded.

Reference: Buzzard et al., *Plug-and-Play Unplugged: Optimization-Free Reconstruction
Using Consensus Equilibrium*, SIAM Journal on Imaging Sciences 11(3), 2018,
https://doi.org/10.1137/17M1122451.

## Scope

The cited papers do not provide unbounded space-time wind generators. These are explicit
adaptations inside the project's existing canonical atlas. They should not be described
as exact reproductions of the source methods, and their default hyperparameters are
untuned pilot settings.
