# Federated Multi-institutional Tumor Segmentation on Multimodal PET-CT Images
sh headfl.sh
# Project Overview
In the domain of AI-driven medical image analysis, our work underscores the pivotal role of multimodal data, particularly in the context of PET-CT imaging. Through FedPT, we have showcased how the integration of PET and CT information, coupled with privacy-preserving federated learning, can significantly enhance diagnostic accuracy and inform clinical decision-making processes. This project opens avenues for further advancements in automated tumor segmentation and paves the way for more robust, privacy-aware AI solutions in healthcare.
# Solution: FedPT
To address these challenges, we propose FedPT, a federated learning approach for jointly training a tumor segmentation model based on PT images from diverse sites, all while safeguarding the privacy of local data. FedPT introduces two key components:

Local Self-Attention (LSA) Module: This module enhances feature representation to effectively fuse PET-CT information. By incorporating local self-attention mechanisms, FedPT ensures that crucial details are preserved during the fusion process.

Adversarial Noise Perturbation (ANP) Module: Designed to operate in heterogeneous federated settings, this module applies adversarial noise perturbations to the fused PT images. This process enhances the robustness of the model while respecting data privacy.
# Dataset
HNSCC：https://wiki.cancerimagingarchive.net/display/Public/HNSCC
WB-FDG：https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287
