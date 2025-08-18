# Xenoverse: Toward Training General-Purpose Learning Agents (GLA) with Randomized Worlds

## xenoverse instead of a single universe

The recent research indicates that the generalization ability of learning agents is primarily dependent on the diversity of training environments. However, the real-world poses a significant limitation on the diversity itself, e.g., physical laws, the gravitational constant is almost constant. We believe this limitation is serious bottleneck to incentivize artificial general intelligence (AGI).

Xenoverse is a collection of extremely diverse worlds by procedural generation based on completely random parameters. We propose that AGI should not be trained and adapted in a single universe, but in xenoverse.

## collection of xenoverse environments

- [AnyMDP](xenoverse/anymdp): Procedurally generated unlimited general-purpose Markov Decision Processes (MDP) in discrete spaces.

- [AnyMDPv2](xenoverse/anymdpv2): Procedurally generated unlimited general-purpose Markov Decision Processes (MDP) in continuous spaces.

- [MetaLanguage](xenoverse/metalang): Pseudo-language generated from randomized neural networks, benchmarking in-context language learning (ICLL).

- [MazeWorld](xenoverse/mazeworld): Procedurally generated immersed 3D mazes with diverse maze structures.

# Installation

```bash
pip install xenoverse
```

# Reference
Related works
```bibtex
@article{wang2024benchmarking,
  title={Benchmarking General Purpose In-Context Learning},
  author={Wang, Fan and Lin, Chuan and Cao, Yang and Kang, Yu},
  journal={arXiv preprint arXiv:2405.17234},
  year={2024}
}
@article{wang2025towards,
  title={Towards Large-Scale In-Context Reinforcement Learning by Meta-Training in Randomized Worlds},
  author={Wang, Fan and Shao, Pengtao and Zhang, Yiming and Yu, Bo and Liu, Shaoshan and Ding, Ning and Cao, Yang and Kang, Yu and Wang, Haifeng},
  journal={arXiv preprint arXiv:2502.02869},
  year={2025}
}
```
