---
title: Low Carbon Footprint for 1D-CNNs with Temporal Max-Pooling
description: Selly Beyene, Sean Giroux, Ria Jayaram, Caleb Lefcort, Ryan Mateo​
layout: libdoc_page.liquid
permalink: index.html
tags:
    - widgets
---

# Introduction
The paper “Low Carbon Footprint for 1D-CNNs with Temporal Max-Pooling” by Anandharaju Durai Raju and Ke Wang addresses growing concerns about the significant carbon emissions associated with AI technologies and proposes solutions to make 1D-CNNs with temporal max-pooling more carbon-efficient. The authors state that AI has become such a high-energy consumer that it emits about as much carbon dioxide as the aviation industry. Current techniques for training CNNs consume substantial GPU memory, resulting in a large carbon footprint. Another issue is that the accessibility of training complex CNNs for researchers is low due to the high cost of high-end GPUs with large amounts of memory.

# Research Questions
This research investigates methods to mitigate the environmental impact of Artificial Intelligence by addressing the high GPU memory consumption and carbon emissions associated with training 1D-CNNs on large sequential inputs. The study explores whether the inherent sparsity mechanism of temporal max-pooling, where only a single "hot" activation per filter contributes to the output, can be leveraged to retroactively prune unused "cold" activations and gradients during backpropagation. Consequently, the core inquiry is whether this targeted strategy can significantly lower computational resource demands and carbon footprints compared to existing memory-efficient variants, such as MalConv2, while ensuring the model achieves identical performance to full computation methods without compromising input sequence length or accuracy.

# Related Work

[2] Raff et al. (MalConv2) — Constant-memory training for extreme-length 1D sequences

Raff et al. introduce a reformulation of temporal max pooling that makes training memory effectively invariant to sequence length, motivated by malware byte sequence classification where inputs can be extremely large. This enables CNN-based malware models to scale to much longer sequences and reports large practical efficiency gains, reframing long-sequence 1D-CNNs as feasible rather than purely memory-bound. Connection to the primary paper: Raju & Wang treat MalConv2 as the closest baseline and then target what remains expensive in practice—especially overhead/complexity and additional forward/backward-pass costs—by exploiting max-pooling sparsity more directly (keeping only the hot activations/gradients instead of full dense maps).

[3] Dao et al. (FlashAttention) — Exact, IO-aware kernel efficiency

FlashAttention shows you can keep attention exact while reducing memory reads/writes by making the algorithm IO-aware (tiling to reduce costly traffic between GPU high-bandwidth memory and on-chip SRAM). It’s a canonical example that major efficiency wins can come from operator/kernel design, not only changing model architecture or using approximations. Connection to the primary paper: HotConv uses the same structure-aware, keep-it-exact philosophy, but aimed at temporal max-pooling in 1D-CNNs—leveraging the fact that only max locations matter for learning to prune cold activations/gradients and cut GPU memory/time without sacrificing accuracy.

[4] Gu et al. (GreenFlow) — Carbon-aware scheduling at the systems level

GreenFlow proposes a GPU-cluster scheduler that reduces job completion time under a carbon-emissions budget, i.e., it optimizes when/where jobs run (and with what configurations) to manage emissions at the cluster level without changing the model itself. Connection to the primary paper: HotConv reduces the intrinsic per-job cost (GPU memory and training time) by exploiting max-pooling sparsity and improving data loading, while GreenFlow reduces emissions through systems scheduling—they’re complementary levers (algorithmic efficiency vs. operational policy).

# Methods 

**PDL (Partitioned Data Loading)**
PDL reduces training time by grouping samples of similar length into partitions stored on disk. During training, only one partition is loaded into RAM at a time, allowing batches with minimal padding while still preserving randomness. This eliminates costly disk access and roughly halves total training time compared to MalConv2.
![Image](/assets/fig3.png)

**HotConv**
A training strategy that reduces GPU memory usage by exploiting the sparsity in temporal max-pooling: only one “hot” activation per filter is needed. It avoids the expensive second forward pass used in MalConv2 by storing only the relevant activations and gradients, reducing GPU memory consumption and carbon footprint without hurting accuracy.
![Image](/assets/alg1.png)

**HotConvEco**
Extends HotConv by also slicing the backward so that only a subset of necessary embeddings and gradients are kept in GPU memory at once. This further reduces GPU memory usage to about 1/22 that of MalConv2, making training faster, more carbon-efficient, and feasible even under tiny GPU budgets.

# Results/Insights
The experimental evaluation demonstrates that the proposed HotConv and HotConvEco approaches significantly outperform existing methods in resource efficiency without compromising model performance.  On the BODMAS dataset, the HotConvEco variant reduced GPU memory consumption to 1/22 of that used by the state-of-the-art MalConv2, while also requiring less training time. When combined with the novel Partitioned Data Loading (PDL) strategy, the approach reduced the overall carbon footprint to approximately 1/7th of MalConv2's footprint and cut training time by half. Critically, these efficiency gains allowed for the use of larger batch sizes and filter counts under constrained budgets, which actually boosted the model's Precision-Recall AUC compared to the baseline. The results confirm that harnessing activation sparsity allows for the processing of extreme-length sequences (up to full sequence length) that would otherwise cause Out-Of-Memory errors in standard MalConv training, proving that sustainable AI training is achievable without sacrificing accuracy.
# Critique & Discussion
## Strengths

Provides memory and energy reductions without performance loss
A major strength of HotConv and HotConvEco is that they achieve significant memory and energy savings without sacrificing model performance. By leveraging the sparsity created by temporal max-pooling, they avoid storing or backpropagating unnecessary activations, significantly reducing memory overhead. This allows for faster training emitting fewer carbon emissions while maintaining identical outputs. This approach does not approximate or modify the model’s behaviour, distinguishing it from several other compression techniques.

Enables easy adoption for existing CNN pipelines
Given that HotConv modifies the training procedures rather than the neural architecture itself, convolutional layers, pooling structure, and downstream classifiers remain unchanged. As a result, existing 1D CNN pipelines can adopt the method with minimal code modifications. This lowers the integration barrier for those who do not wish to redesign models from scratch. In fields like malware analysis or genomics, where models are already well-established, this can be an especially key benefit. 

## Weaknesses/Limitations

Restricted to architectures using temporal max-pooling
A key limitation of HotConvEco is that its effectiveness depends on the behaviour of temporal max-pooling. In max-pooling, only the position with the highest activation contributes to the final output, meaning that only this position receives a meaningful gradient during backpropagation. Due to this property, the method can ignore all other activations and drastically reduce memory usage. Models that use average pooling, attention mechanisms, or recurrent layers do not exhibit this pattern as gradients are distributed across higher numbers of positions. As a result, the approach used by HotConvEco only applies to a subset of 1D CNN architectures. In particular, it may become less broadly applicable as sequence models shift further toward transformer-based designs.

Limited gains from partitioned data loading on highly varied datasets
Partitioned Data Loading (PDL) aims to reduce padding and I/O overhead by grouping inputs of similar length. This can be very effective when input sizes fall within a reasonable range. However, datasets with very high variability in input length may still produce partitions with hefty size differences, limiting the reduction needed in padding. In these cases, the computation required for the additional preprocessing steps may not be justified. 

## Ethical/global implications

Reduces hardware barrier for long-sequence models
Long-sequence models often require substantial GPU memory, which can make them inaccessible to smaller institutions or research groups with limited computational resources. HotConvEvo helps remove this barrier by enabling long-sequence training on more constrained hardware, including older GPUs with limited memory. This increased accessibility allows a wider range of the computer science community to work with complex sequence tasks that would be infeasible otherwise. 

Lowers the environmental footprint of training
By both reducing memory usage and shortening training time, HotConvEco lowers the overall carbon emissions associated with each training cycle. As the volume of data required to train machine learning models continues to increase, these factors become especially important. Through consuming fewer resources while maintaining comparable performance, HotConvEvo sets the stage for more sustainable AI/ML research and deployment.


# Future Directions
Extend to neural architectures beyond max-pooling
Given that only a subset of models rely on temporal max-pooling, one direction for future work could be the exploration of how the ideas behind this innovation can be adapted to models based on other pooling operations. Sequence architectures based on operations like average pooling and attention distribute gradients across multiple positions rather than a single one. Identifying forms of sparsity within these architectures could help improve the robustness of these findings. 
Evaluate top-k pooling variants to reduce noise sensitivity
While the sparsity generated by max-pooling heavily reduces resource use, it depends on just a single activation. This can make models very sensitive to noise or fluctuations in the input. Introducing the option to utilize k activations as opposed to one could combat this sensitivity while maintaining substantial levels of sparsity. 
Validate performance on a more diverse range of long-sequence tasks
This paper focuses on malware classification, and seems theoretically promising for 1D CNNs using temporal max-pooling for other applications. However, it is important that empirical validation data confirms this. Applying this method to domains like genomics could offer a more complete view of its strengths and limitations. This broader validation would also ensure that its memory and energy reduction benefits remain consistent across tasks with different structures.

# References
[1] Anandharaju Durai Raju and Ke Wang. 2024. Low Carbon Footprint Training for 1D-CNNs with Temporal Max-Pooling. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, Boise, ID, USA, October 21–25. ACM, New York, NY, USA, 549–559. DOI: 10.1145/3627673.3679678

[2] Edward Raff, William Fleshman, Richard Zak, Hyrum S. Anderson, Bobby Filar, and Mark McLean. 2021. Classifying Sequences of Extreme Length with Constant Memory Applied to Malware Detection. Proceedings of the AAAI Conference on Artificial Intelligence 35, 11 (2021), 9386–9394. DOI: 10.1609/aaai.v35i11.17131

[3] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. In Advances in Neural Information Processing Systems, Vol. 35. arXiv:2205.14135. DOI: 10.48550/arXiv.2205.14135

[4] Di Gu, Yujia Zhao, Peng Sun, Xiaoyang Jin, and Xu Liu. 2025. GreenFlow: A Carbon-Efficient Scheduler for Deep Learning Workloads. IEEE Transactions on Parallel and Distributed Systems 36, 2 (2025), 168–184. DOI: 10.1109/TPDS.2024.3470074

[5] Nestor Maslej, Loredana Fattorini, Erik Brynjolfsson, John Etchemendy, Katrina Ligett, Terah Lyons, James Manyika, Helen Ngo, Juan Carlos Niebles, Vanessa Parli, Yoav Shoham, Russell Wald, Jack Clark, and Raymond Perrault. 2023. Artificial Intelligence Index Report 2023. Stanford Institute for Human-Centered Artificial Intelligence (HAI), Stanford University. arXiv:2310.03715. DOI: 10.48550/arXiv.2310.03715

[6] Nestor Maslej, Loredana Fattorini, Raymond Perrault, Yolanda Gil, Vanessa Parli, Njenga Kariuki, Emily Capstick, Anka Reuel, Erik Brynjolfsson, John Etchemendy, Katrina Ligett, Terah Lyons, James Manyika, Juan Carlos Niebles, Yoav Shoham, Russell Wald, Tobi Walsh, Armin Hamrah, Lapo Santarlasci, Julia Betts Lotufo, Alexandra Rome, Andrew Shi, and Sukrut Oak. 2025. Artificial Intelligence Index Report 2025. Stanford Institute for Human-Centered Artificial Intelligence (HAI), Stanford University. arXiv:2504.07139. DOI: 10.48550/arXiv.2504.07139
