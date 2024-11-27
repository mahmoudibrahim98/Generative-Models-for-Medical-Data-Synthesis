# **Generative Models for Medical Data Synthesis**

Welcome to the repository for the survey paper, **"Generative Models for Medical Data Synthesis: A Systematic Review"**. This repository provides links to the papers and code referenced in the survey, along with detailed summaries of the evaluation methods used across various data modalities.

## **Introduction**

Generative models such as GANs, VAEs, Diffusion Models, and LLMs have revolutionized the synthesis of medical data, including:
- **EHR (Electronic Health Records)** for tabular data.
- **Signals** such as ECG and PPG.
- **Imaging** data, including dermoscopic, mammographic, ultrasound, CT, MRI, and X-ray images.
- **Text** for clinical notes and radiology reports.

This repository is structured to:
- Provide **easy access** to the papers and code repositories.
- Highlight **evaluation methods** for generative models in each modality.

---

## **Table of Contents**

1. [Overview of Tables](#overview-of-tables)
2. [Electronic Health Records (EHR)](#electronic-health-records-ehr)
3. [Signals](#signals)
   - [Electrocardiogram (ECG)](#electrocardiogram-ecg)
   - [Other Signals](#other-signals)
4. [Images](#images)
   - [Dermoscopic Images](#dermoscopic-images)
   - [Mammographic Images](#mammographic-images)
   - [Other Imaging Modalities](#other-imaging-modalities)
5. [Text](#text)
6. [Evaluation Metrics and Techniques](#evaluation-metrics-and-techniques)
7. [Contributing](#contributing)
8. [License](#license)

---

## **Overview of Tables**

The repository contains 12 tables categorized as follows:

| **Table Number** | **Category**      | **Modality**         |
|-------------------|-------------------|----------------------|
| 1                 | Tabular Data      | Electronic Health Records (EHR) |
| 2-3               | Signals           | ECG, PPG, and other physiological signals |
| 4-11              | Images            | Dermoscopic, mammographic, ultrasound, CT, MRI, X-ray, and other imaging |
| 12                | Text              | Clinical notes and radiology reports |

Each table includes the **application, model type, technology, paper links, code repositories**, and **evaluation methods**.

---

## **Electronic Health Records (EHR)**

### Electronic Health Records

| **Application**            | **Type**   | **Technology**      | **Paper**                               | **Code**                                | **Evaluation**            |
|-----------------------------|------------|----------------------|-----------------------------------------|-----------------------------------------|---------------------------|
| Patient Demographics Gen.  | GAN        | Tabular GAN          | [Paper Link](https://example.com)      | [Code Link](https://github.com/example) | Fidelity (MSE)            |
| Disease Progression Model  | VAE        | Bayesian VAE         | [Paper Link](https://example.com)      | [Code Link](https://github.com/example) | Fidelity (KL Divergence), Privacy (k-Anonymity) |

---

## **Signals**

### Electrocardiogram (ECG)
| Type    | Application                                           | Architecture                                           | Paper Link                                                                            | Code Link                                                                                                   | Evaluation   | Date    |
|:--------|:------------------------------------------------------|:-------------------------------------------------------|:--------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|:-------------|:--------|
| GAN     | Intra-translation                                     | Bi-LSTM and CNN                                        | [Paper Link](https://arxiv.org/abs/2310.03753)                                        | [Code Link](https://github.com/maxbagga/ScarletEagle1)                                                      | F            | 2023-09 |
| DM      | Class-conditional                                     | DSAT-ECG                                               | [Paper Link](https://doi.org/10.3390/s23198328)                                       |                                                                                                             | U, F         | 2023-09 |
| Other   | Unconditonal                                          | Bi-LSTM and CNN                                        | [Paper Link](https://doi.org/10.1007/s13755-023-00241-y)                              |                                                                                                             | U            | 2023-08 |
| DM      | Inter-translation                                     | Region-Disentangled Diffusion Model (RDDM)             | [Paper Link](https://arxiv.org/abs/2308.13568)                                        | [Code Link](https://github.com/DebadityaQU/RDDM)                                                            | U, F, C      | 2023-08 |
| DM      | conditioning on other ECG statements;prioir knowledge | SSSD-ECG                                               | [Paper Link](https://doi.org/10.1016/j.compbiomed.2023.107115)                        | [Code Link](https://github.com/AI4HealthUOL/SSSD-ECG)                                                       | U, Q         | 2023-06 |
| DM      | Class-conditional                                     | DDPM-based:DiffECG                                     | [Paper Link](https://arxiv.org/abs/2306.01875)                                        |                                                                                                             | U, F, Q      | 2023-06 |
| GAN     | Intra-translation                                     | StarGAN v2                                             | [Paper Link](https://arxiv.org/pdf/2103.00006)                                        | [Code Link](Original method)                                                                                | U, Q         | 2023-06 |
| DM      | Unconditonal                                          | image-based:DDPM                                       | [Paper Link](https://arxiv.org/pdf/2303.02475)                                        | [Code Link](https://github.com/mah533/Synthetic-ECG-Signal-Generation-using-Probabilistic-Diffusion-Models) | U, F         | 2023-05 |
| GAN     | Unconditonal                                          | LSTM-based:TS-GAN                                      | [Paper Link](https://doi.org/10.1145/3583593)                                         |                                                                                                             | U, F, Q      | 2023-04 |
| GAN,VAE | Class-conditional                                     | CVAE,CWGAN                                             | [Paper Link](https://doi.org/10.1016/j.bspc.2023.104587)                              |                                                                                                             | U            | 2023-04 |
| VAE,GAN | text-to-signal                                        | Auto-TTE                                               | [Paper Link](https://arxiv.org/abs/2303.09395)                                        | [Code Link](https://github.com/TClife/text_to_ecg)                                                          | U, F, D, Q   | 2023-03 |
| GAN,AE  | Inter-translation                                     | classical GAN,adversarial AE,modality transfer GAN     | [Paper Link](https://doi.org/10.1109/JBHI.2022.3223777)                               |                                                                                                             | U, F, Q      | 2023-02 |
| GAN     | Class-conditional                                     | WGAN-GP-based:AC-WGAN-GP                               | [Paper Link](https://arxiv.org/abs/2202.00569)                                        | [Code Link](https://github.com/mah533/Augmentation-of-ECG-Training-Dataset-with-CGAN)                       | U            | 2022-11 |
| GAN     | Clinical Knowledge                                    | WGAN-GP-based:CardiacGen                               | [Paper Link](https://arxiv.org/abs/2211.08385)                                        | [Code Link](https://github.com/SENSE-Lab-OSU/cardiac_gen_model)                                             | U, F, C, P   | 2022-11 |
| GAN     | Unconditonal                                          | classic GAN , DC-DC GAN , BiLSTM-DC , AE/VAE-DC , WGAN | [Paper Link](https://arxiv.org/abs/2112.03268)                                        | [Code Link](https://github.com/mah533/Synthetic-ECG-Generation---GAN-Models-Comparison)                     | U, F, Q, C   | 2022-08 |
| GAN     | Unconditonal                                          | image-based:TTS-GAN                                    | [Paper Link](https://doi.org/10.1007/978-3-031-09342-5_13)                            | [Code Link](https://github.com/imics-lab/tts-gan)                                                           | F, Q         | 2022-06 |
| GAN     | conditioning on other ECG statements;prioir knowledge | Conditional GAN                                        | [Paper Link](https://doi.org/10.1145/3477314.3507300)                                 |                                                                                                             | U, Q         | 2022-05 |
| VAE     | specific subject characteristics                      | cVAE                                                   | [Paper Link](https://doi.org/10.1109/ISBI52829.2022.9761431)                          |                                                                                                             | F            | 2022-04 |
| GAN,VAE | Class-conditional                                     | PHYSIOGAN                                              | [Paper Link](https://arxiv.org/abs/2204.13597)                                        | [Code Link](yES)                                                                                            | U, F, D      | 2022-04 |
| GAN     | Class-conditional                                     | DCCGAN (Deep convolutional condtional GAN)             | [Paper Link](https://doi.org/10.1007/978-3-030-91390-8_12)                            |                                                                                                             | U, F, Q      | 2022-02 |
| GAN     | Unconditonal                                          | WaveGAN,Pulse2Pulse                                    | [Paper Link](https://doi.org/10.1038/s41598-021-01295-2)                              | [Code Link](https://github.com/vlbthambawita/deepfake-ecg)                                                  | P            | 2021-11 |
| GAN     | Unconditonal                                          | Composite GAN:LSTM-GAN and DCGAN                       | [Paper Link](https://doi.org/10.23919/EUSIPCO54536.2021.9616079)                      |                                                                                                             | U            | 2021-08 |
| GAN     | Unconditonal                                          | LSTM-based:BiLSTM                                      | [Paper Link](https://www.sciencedirect.com/science/article/abs/pii/S0022073621001850) |                                                                                                             | F            | 2021-06 |
### Other Signals

| **Application**         | **Type**   | **Modality**     | **Technology**  | **Paper**                           | **Code**                                | **Evaluation**            |
|--------------------------|------------|------------------|------------------|-------------------------------------|-----------------------------------------|---------------------------|
| Physiological Signal Gen | GAN        | Multi-signal     | AC-WGAN-GP       | [Paper Link](https://example.com)  | [Code Link](https://github.com/example) | Fidelity (MSE), Diversity (Variance) |

---

## **Images**

This section includes papers for dermoscopic, mammographic, ultrasound, CT, MRI, X-ray, and multi-modal imaging data.

### Table 4: Dermoscopic Images

| **Application**         | **Type**   | **Modality**   | **Technology**  | **Paper**                           | **Code**                                | **Evaluation**                |
|--------------------------|------------|----------------|------------------|-------------------------------------|-----------------------------------------|-------------------------------|
| Lesion Segmentation      | GAN        | Dermoscopic    | DCGAN            | [Paper Link](https://example.com)  | [Code Link](https://github.com/example) | Fidelity (IoU), Diversity (IS), Clinical Review |

---

## **Text**

### Table 12: Clinical Notes and Radiology Reports

| **Application**          | **Type**       | **Technology**         | **Paper**                           | **Code**                                | **Evaluation**                |
|---------------------------|----------------|-------------------------|-------------------------------------|-----------------------------------------|-------------------------------|
| Clinical Note Gen.        | LLM           | GPT-3                  | [Paper Link](https://example.com)  | [Code Link](https://github.com/example) | Language Coherence (BLEU), Privacy (k-Anonymity) |

---

## **Evaluation Metrics and Techniques**

Generative models in medical data synthesis are evaluated using a variety of metrics:

1. **Fidelity (F)**:
   - Fidelity of synthetic data to real data, measured using metrics like:
     - MSE, IoU, Dice Score, PSNR.
     - Expert review for clinical validity.
2. **Diversity (D)**:
   - Diversity of generated data using metrics like:
     - FID, Inception Score (IS), Variance measures.
3. **Clinical Validity (C)**:
   - Reviewed by domain experts for clinical significance and utility.
4. **Privacy (P)**:
   - Privacy-preserving evaluation methods such as:
     - k-Anonymity, Differential Privacy (DP).

---

## **Contributing**

We welcome contributions to keep this repository updated with new research and implementations. You can contribute by:
- Adding new papers or implementations.
- Proposing enhancements to the structure or content.
- Submitting pull requests for corrections.

Please follow our [Contribution Guidelines](CONTRIBUTING.md).

---

## **License**

This repository is licensed under the [MIT License](LICENSE).

---

## **Acknowledgments**

We thank all researchers and practitioners contributing to advancements in medical data synthesis through generative models. This repository is part of our commitment to promote collaboration and open research in this field.
