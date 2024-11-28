# **Generative Models for Medical Data Synthesis**

Welcome to the repository for the survey paper, [**"Generative Models for Medical Data Synthesis: A Systematic Review"**](https://arxiv.org/abs/2407.00116). This repository provides links to the papers and code referenced in the survey, along with detailed summaries of the evaluation methods used across various data modalities.

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

1. [Synthesis applications and purpose of synthesis](#synthesis-applications)
2. [Electronic Health Records (EHR)](#electronic-health-records-ehr)
3. [Signals](#signals)
   - [Electrocardiogram (ECG)](#electrocardiogram-ecg)
   - [Electroencephalogram (EEG)](#other-signals)
4. [Images](#images)
   - [Dermoscopic Images](#dermoscopic-images)
   - [Mammographic Images](#mammographic-images)
   - [Ultrasound  Images](#ultrasound-images)
   - [MRI  Images](#mri-images)
   - [CT  Images](#ct-images)
   - [X-ray  Images](#xray-images)
   - [OCT  Images](#oct-images)
   - [Multiple modalities Images](#multiple-imaging-modalities)
5. [Text](#text)
6. [Evaluation Metrics and Techniques](#evaluation-metrics-and-techniques)
7. [Contributing](#contributing)
8. [License](#license)

---

## **Synthesis applications and purpose of synthesis**
Generative models in medical data synthesis can be broadly categorized into unconditional and conditional models:

1. Unconditional Models: These models take a random variable as input and generate data without additional context or guidance.
2. Conditional Models: These models incorporate external information, such as images, text, semantic maps, class labels, attributes, or signals, to guide the generation process.

### **Key Synthesis Applications**

| **Data Type**            | **Synthesis Application**       | **Description**                                                                                     | **Examples**                                                                                       |
|---------------------------|----------------------------------|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **EHR**                  | Longitudinal EHR                | Medical codes from multiple patient visits.                                                        | Patient diagnostic history across hospital visits.                                                |
|                           | Aggregated EHR                 | Longitudinal data condensed into a single row.                                                     | Summary of all patient visits in a single record.                                                 |
|                           | Time-dependent EHR             | Time-series readings from a single patient visit.                                                  | Vitals recorded during a hospital stay.                                                           |
|                           | Snapshot EHR                   | A single snapshot focusing on specific patient attributes.                                          | Demographic details with selected health metrics.                                                 |
| **Imaging & Signals**     | Inter-modal Translation         | Converts data from one modality to another.                                                        | CT to MRI, ECG to PPG.                                                                            |
|                           | Intra-modal Translation         | Translates data within the same modality.                                                          | T1-weighted MRI to T2-weighted MRI, single-lead ECG to 12-lead ECG.                               |
|                           | Class or Semantic Map Synthesis | Generates data based on class labels or segmentation masks.                                         | Brain MRI with a tumor generated from a tumor mask, ECG labeled as "arrhythmia."                 |
|                           | Attribute-based Synthesis       | Generates data based on patient-specific attributes (e.g., age, sex, BMI).                         | Personalized synthetic brain MRIs or ECG signals.                                                 |
|                           | Text-based Synthesis            | Integrates clinical text into synthetic data generation.                                            | Generating X-rays or ECG signals from textual descriptions like "moderate bilateral pleural effusion." |
| **Text**                 | NLP Enhancement                 | Improves tasks like NER, information extraction, summarization, and question answering.             | Improving NER on clinical notes with synthetic text.                                              |
|                           | Text Augmentation               | Generates additional clinical notes, discharge summaries, or reports.                              | Augmenting patient reports when real data is limited.                                             |
|                           | Text De-identification          | Removes or replaces PII while preserving utility and privacy.                                       | Replacing names, addresses, or diagnoses in clinical notes.                                       |


## **Overview of Tables**

The repository contains 12 tables categorized as follows:

| **Table Number** | **Category**      | **Modality**         |
|-------------------|-------------------|----------------------|
| 1                 | Tabular Data      | Electronic Health Records (EHR) |
| 2-3               | Signals           | ECG, PPG, and other physiological signals |
| 4-11              | Images            | Dermoscopic, mammographic, ultrasound, CT, MRI, X-ray, and other imaging |
| 12                | Text              | Clinical notes and radiology reports |

Each table includes the **application, model type, technology, Paperlinks, code repositories**, and **evaluation methods**.

---

## **Electronic Health Records (EHR)**

### Electronic Health Records

| **Application**            | **Type**   | **Technology**      | **Paper**                               | **Code**                                | **Evaluation**            |
|-----------------------------|------------|----------------------|-----------------------------------------|-----------------------------------------|---------------------------|
| Patient Demographics Gen.  | GAN        | Tabular GAN          | [Paper](https://example.com)      | [Code ](https://github.com/example) | Fidelity (MSE)            |
| Disease Progression Model  | VAE        | Bayesian VAE         | [Paper](https://example.com)      | [Code ](https://github.com/example) | Fidelity (KL Divergence), Privacy (k-Anonymity) |

---

## **Signals**

### Electrocardiogram (ECG)
| Type    | Application                                           | Architecture                                           | Paper Link                                                                       | Code Link                                                                                              | Evaluation   | Date    |
|:--------|:------------------------------------------------------|:-------------------------------------------------------|:---------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------|:-------------|:--------|
| GAN     | Intra-translation                                     | Bi-LSTM and CNN                                        | [arXiv](https://arxiv.org/abs/2310.03753)                                        | [Code](https://github.com/maxbagga/ScarletEagle1)                                                      | F            | 2023-09 |
| DM      | Class-conditional                                     | DSAT-ECG                                               | [Paper](https://doi.org/10.3390/s23198328)                                       |                                                                                                        | U, F         | 2023-09 |
| Other   | Unconditonal                                          | Bi-LSTM and CNN                                        | [Paper](https://doi.org/10.1007/s13755-023-00241-y)                              |                                                                                                        | U            | 2023-08 |
| DM      | Inter-translation                                     | Region-Disentangled Diffusion Model (RDDM)             | [arXiv](https://arxiv.org/abs/2308.13568)                                        | [Code](https://github.com/DebadityaQU/RDDM)                                                            | U, F, C      | 2023-08 |
| DM      | conditioning on other ECG statements;prioir knowledge | SSSD-ECG                                               | [Paper](https://doi.org/10.1016/j.compbiomed.2023.107115)                        | [Code](https://github.com/AI4HealthUOL/SSSD-ECG)                                                       | U, Q         | 2023-06 |
| DM      | Class-conditional                                     | DDPM-based:DiffECG                                     | [arXiv](https://arxiv.org/abs/2306.01875)                                        |                                                                                                        | U, F, Q      | 2023-06 |
| GAN     | Intra-translation                                     | StarGAN v2                                             | [arXiv](https://arxiv.org/pdf/2103.00006)                                        | [Original](https://github.com/clovaai/stargan-v2)                                                                                | U, Q         | 2023-06 |
| DM      | Unconditonal                                          | image-based:DDPM                                       | [arXiv](https://arxiv.org/pdf/2303.02475)                                        | [Code](https://github.com/mah533/Synthetic-ECG-Signal-Generation-using-Probabilistic-Diffusion-Models) | U, F         | 2023-05 |
| GAN     | Unconditonal                                          | LSTM-based:TS-GAN                                      | [Paper](https://doi.org/10.1145/3583593)                                         |                                                                                                        | U, F, Q      | 2023-04 |
| GAN,VAE | Class-conditional                                     | CVAE,CWGAN                                             | [Paper](https://doi.org/10.1016/j.bspc.2023.104587)                              |                                                                                                        | U            | 2023-04 |
| VAE,GAN | text-to-signal                                        | Auto-TTE                                               | [arXiv](https://arxiv.org/abs/2303.09395)                                        | [Code](https://github.com/TClife/text_to_ecg)                                                          | U, F, D, Q   | 2023-03 |
| GAN,AE  | Inter-translation                                     | classical GAN,adversarial AE,modality transfer GAN     | [Paper](https://doi.org/10.1109/JBHI.2022.3223777)                               |                                                                                                        | U, F, Q      | 2023-02 |
| GAN     | Class-conditional                                     | WGAN-GP-based:AC-WGAN-GP                               | [arXiv](https://arxiv.org/abs/2202.00569)                                        | [Code](https://github.com/mah533/Augmentation-of-ECG-Training-Dataset-with-CGAN)                       | U            | 2022-11 |
| GAN     | Clinical Knowledge                                    | WGAN-GP-based:CardiacGen                               | [arXiv](https://arxiv.org/abs/2211.08385)                                        | [Code](https://github.com/SENSE-Lab-OSU/cardiac_gen_model)                                             | U, F, C, P   | 2022-11 |
| GAN     | Unconditonal                                          | classic GAN , DC-DC GAN , BiLSTM-DC , AE/VAE-DC , WGAN | [arXiv](https://arxiv.org/abs/2112.03268)                                        | [Code](https://github.com/mah533/Synthetic-ECG-Generation---GAN-Models-Comparison)                     | U, F, Q, C   | 2022-08 |
| GAN     | Unconditonal                                          | image-based:TTS-GAN                                    | [Paper](https://doi.org/10.1007/978-3-031-09342-5_13)                            | [Code](https://github.com/imics-lab/tts-gan)                                                           | F, Q         | 2022-06 |
| GAN     | conditioning on other ECG statements;prioir knowledge | Conditional GAN                                        | [Paper](https://doi.org/10.1145/3477314.3507300)                                 |                                                                                                        | U, Q         | 2022-05 |
| VAE     | specific subject characteristics                      | cVAE                                                   | [Paper](https://doi.org/10.1109/ISBI52829.2022.9761431)                          |                                                                                                        | F            | 2022-04 |
| GAN,VAE | Class-conditional                                     | PHYSIOGAN                                              | [arXiv](https://arxiv.org/abs/2204.13597)                                        | [Code](yES)                                                                                            | U, F, D      | 2022-04 |
| GAN     | Class-conditional                                     | DCCGAN (Deep convolutional condtional GAN)             | [Paper](https://doi.org/10.1007/978-3-030-91390-8_12)                            |                                                                                                        | U, F, Q      | 2022-02 |
| GAN     | Unconditonal                                          | WaveGAN,Pulse2Pulse                                    | [Paper](https://doi.org/10.1038/s41598-021-01295-2)                              | [Code](https://github.com/vlbthambawita/deepfake-ecg)                                                  | P            | 2021-11 |
| GAN     | Unconditonal                                          | Composite GAN:LSTM-GAN and DCGAN                       | [Paper](https://doi.org/10.23919/EUSIPCO54536.2021.9616079)                      |                                                                                                        | U            | 2021-08 |
| GAN     | Unconditonal                                          | LSTM-based:BiLSTM                                      | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0022073621001850) |                                                                                                        | F            | 2021-06 |
### Electroencephalogram (EEG)

| Type           | Application                   | Architecture                 | Paper Link                                                                                      | Code Link                                                                                                     | Evaluation   | Date    |
|:---------------|:------------------------------|:-----------------------------|:------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------|:-------------|:--------|
| DM             | Unconditional                 | LDM                          | [Paper](https://openreview.net/forum?id=mDwURmlapW)                                             | [Code](https://github.com/bruAristimunha/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models) | F            | 2023-10 |
| DM             | Class-conditional             | DDPM                         | [Paper](https://doi.org/10.1016/j.patter.2024.101047)                                           |                                                                                                               | U, F, C      | 2023-08 |
| DM             | Conditioned STFT spectrograms | DiffEEG                      | [arXiv](http://arxiv.org/abs/2306.08256v1)                                                      |                                                                                                               | U, F         | 2023-06 |
| VAE            | Unconditional                 | causal recurrent CAE (CRVAE) | [arXiv](http://arxiv.org/abs/2301.06574v1)                                                      | [Code](https://github.com/hongmingli1995/CR-VAE)                                                              | U, F, Q      | 2023-01 |
| GAN            | Class-conditional             | Conditional Wasserstein GAN  | [Paper](https://doi.org/10.1155/2022/7028517)                                                   |                                                                                                               | U            | 2022-03 |
| GAN            | Unconditional                 | temporal GAN(TGAN)           | [Paper](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2022.755094/full) |                                                                                                               | U, Q         | 2022-02 |
| Language Model | Unconditional                 | GPT2                         | [Paper](https://doi.org/10.1109/LRA.2021.3056355)                                               | [Code](https://github.com/jordan-bird/Generational-Loop-GPT2)                                                 | U            | 2021-02 |

## **Images**

This section includes papers for dermoscopic, mammographic, ultrasound, CT, MRI, X-ray, and multi-modal imaging data.

### Dermoscopic Images
| Type            | Application                     | Architecture                                        | Paper Link                                                | Code Link                                                     | Evaluation    | Date    |
|:----------------|:--------------------------------|:----------------------------------------------------|:----------------------------------------------------------|:--------------------------------------------------------------|:--------------|:--------|
| GAN             | Unconditional                   | Pgan                                                | [Paper](https://doi.org/10.1016/j.artmed.2023.102556)     |                                                               | U, F          | 2023-10 |
| Diffusion Model | text-to-image                   | LDM, Stable Diffusion , Fine tuned stable diffusion | [arXiv](http://arxiv.org/abs/2308.12453v1)                |                                                               | U             | 2023-08 |
| GAN             | Unconditional                   | StyleGAN2-ADA                                       | [arXiv](http://arxiv.org/abs/2303.04839v1)                | [Code](https://github.com/thinkercache/stylegan2-ada-pytorch) | F, Q          | 2023-03 |
| Diffusion Model | text-to-image                   | LDM                                                 | [arXiv](https://arxiv.org/abs/2301.04802)                 |                                                               | U, Q          | 2023-01 |
| Diffusion Model | text-to-image                   | DALL-E2                                             | [arXiv](http://arxiv.org/abs/2211.13352v1)                |                                                               | U             | 2022-11 |
| GAN             | majority to minority conversion | CycleGAN                                            | [Paper](https://doi.org/10.1002/ima.22880)                |                                                               | U, F          | 2022-09 |
| GAN             | Class conditional               | StyleGAN2-ADA                                       | [Paper](https://doi.org/10.1007/978-3-031-16452-1_42)     | [Code](https://github.com/AgaMiko/debiasing-effect-of-gans.)  | U, F          | 2022-09 |
| GAN             | Class conditional               | StyleGAN2-ADA                                       | [arXiv](https://arxiv.org/abs/2208.11702)                 | [Code](https://github.com/aidotse/stylegan2-ada-pytorch)      | U, F, D, Q, P | 2022-08 |
| GAN             | Unconditional                   | StyleGAN2                                           | [Paper](https://doi.org/10.1007/s11265-022-01757-4)       |                                                               | U, F          | 2022-04 |
| GAN             | Class conditional               | cGAN                                                | [Paper](https://doi.org/10.1109/TENCON54134.2021.9707291) |                                                               | U             | 2021-12 |
| GAN             | Unconditional                   | SLA-StyleGAN                                        | [Paper](https://doi.org/10.1109/ACCESS.2021.3049600)      |                                                               | U, F, Q       | 2021-01 |


### Mammography Images

| Type            | Application       | Architecture                         | Paper Link                                                 | Code Link                                                             | Evaluation   | Date    |
|:----------------|:------------------|:-------------------------------------|:-----------------------------------------------------------|:----------------------------------------------------------------------|:-------------|:--------|
| GAN             | Class conditional | Cgan                                 | [Paper](https://doi.org/10.1016/j.bspc.2023.105456)        |                                                                       | U            | 2024-01 |
| Diffusion Model | text-to-image     | FineTuned_StableDiffusion            | [Paper](https://doi.org/10.3390/s24072076)                 | [Code](https://github.com/Likalto4/diffusion-models_master/tree/main) | F, Q         | 2023-06 |
| GAN             | Intra-translation | CycleGAN                             | [Paper](https://doi.org/10.3389/fonc.2022.1044496)         | [Code](https://github.com/RichardObi/medigan)                         | U, F, Q, P   | 2023-01 |
| GAN             | Intra-translation | complete representation GAN (CR-GAN) | [Paper](https://doi.org/10.3390/app122312206)              | [Original](https://github.com/bluer555/CR-GAN)           | F, P         | 2022-11 |
| GAN             | Intra-translation | Pix2Pix                              | [Paper](https://doi.org/10.1109/IMPACT55510.2022.10029087) |                                                                       | U, F, P      | 2022-11 |
| GAN             | Intra-translation | pGAN variant                         | [Paper](https://doi.org/10.1016/j.compbiomed.2022.105444)  |                                                                       | U            | 2022-06 |
| GAN             | Class conditional | ROImammoGAN                          | [Paper](https://doi.org/10.1038/s41598-022-09929-9)        |                                                                       | F, P         | 2022-04 |
| GAN             | Intra-translation | HRGAN, based on CycleGAN             | [Paper](https://doi.org/10.3389/fonc.2022.868257)          |                                                                       | U            | 2022-04 |
| GAN             | Unconditional     | DCGAN,WGAN-GP                        | [Paper](https://doi.org/10.1117/12.2625781)                |                                                                       | U, P         | 2022-03 |
| GAN             | Intra-translation | Pix2Pix                              | [Paper](https://doi.org/10.1109/TMI.2021.3108949)          |                                                                       | U, P         | 2021-12 |
| GAN             | Intra-translation | DCGAN , InfillingGAN                 | [Paper](https://doi.org/10.1016/j.cmpb.2021.106019)        | [Original](https://github.com/Natsu6767/DCGAN-PyTorch)                                      | U, F, P      | 2021-04 |

### Ultrasound Images

| Type                    | Application       | Architecture       | Paper Link                                               | Code Link                                                    | Evaluation   | Date    |
|:------------------------|:------------------|:-------------------|:---------------------------------------------------------|:-------------------------------------------------------------|:-------------|:--------|
| GAN                     | inter-Translation | ApGAN              | [Paper](https://doi.org/10.1016/j.cmpb.2023.107642)      |                                                              | U, F, Q, C   | 2023-10 |
| VAE                     | inter-Translation | MHVAE              | [Paper](https://doi.org/10.1007/978-3-031-43999-5_43)    | [Code](https://github.com/ReubenDo/MHVAE)                    | F            | 2023-10 |
| GAN                     | Class conditional | GAN-CA             | [Paper](https://doi.org/10.1109/ICET58434.2023.10211926) | [Code](https://github.com/jalbayd1/myopathy_US)              | U            | 2023-08 |
| gan                     | Class conditional | Phased GAN         | [Paper](https://doi.org/10.1016/j.bspc.2023.105196)      |                                                              | U            | 2023-07 |
| Diffusion Models        | Class conditional | DDPM               | [Paper](https://doi.org/10.1007/978-3-031-44521-7_4)     | [Code](https://github.com/david-stojanovski/echo_from_noise) | U            | 2023-05 |
| Diffusion Model and GAN | Unconditional     | DSR-GAN,TB-GAN     | [arXiv](http://arxiv.org/abs/2304.03941v1 )              | [Code](https://github.com/budai4medtech/midl2023)            | F            | 2023-04 |
| gan                     | intra-translation | U-net based gen    | [Paper](https://doi.org/10.1038/s41467-023-36102-1)      | [Code](https://github.com/yyyzzzhao/VEUS)                   | U, F, Q      | 2023-02 |
| GAN                     | inter-Translation | CycleGAN based     | [Paper](https://doi.org/10.1109/IUS54386.2022.9957451)   |                                                              | U            | 2022-12 |
| GAN                     | inter-Translation | 3D Pix2pix         | [Paper](https://doi.org/10.1109/ACCESS.2022.3207177)     |                                                              | U, Q         | 2022-09 |
| GAN                     | Unconditional     | StyleGAN2 variants | [Paper](https://doi.org/10.1515/cdbme-2022-0005)         |                                                              | U, F         | 2022-07 |
| gan                     | intra-translation | spGAN              | [Paper](https://doi.org/10.1016/j.media.2022.102461)     | [Code](https://github.com/Carmenliang/UI_synthesis)          | U, F         | 2022-04 |
| GAN                     | inter-Translation | CycleGAN           | [Paper](https://doi.org/10.1016/j.ultras.2022.106706)    |                                                              | U, F         | 2022-02 |
| gan                     | intra-translation | pix2pix based      | [Paper](https://doi.org/10.3390/diagnostics12020253)     |                                                              | F, Q         | 2022-01 |
| gan                     | intra-translation | PSFFGAN            | [Paper](https://doi.org/10.1109/JBHI.2022.3143319)       |                                                              | F, Q         | 2022-01 |
| H:GAN,VAE               | Unconditional     | Improved α-WGAN-GP | [Paper](https://doi.org/10.1109/IUS52206.2021.9593403)   |                                                              | U, F         | 2021-11 |
| GAN                     | Unconditional     | stylegan2-ada      | [Paper](https://doi.org/10.3390/s21237975)               | [Code](https://github.com/albertoMontero/stylegan2-ada)      | U, F         | 2021-11 |
| GAN                     | Unconditional     | StackGAN           | [Paper](https://doi.org/10.1007/978-3-030-87583-1_18)    |                                                              | U, F, Q      | 2021-07 |
| GAN                     | Unconditional     | TripleGAN          | [Paper](https://doi.org/10.1016/j.cmpb.2021.106018)      |                                                              | U, Q         | 2021-02 |

### MRI Images

| **Application**         | **Type**   | **Modality**   | **Technology**  | **Paper**                           | **Code**                                | **Evaluation**                |
|--------------------------|------------|----------------|------------------|-------------------------------------|-----------------------------------------|-------------------------------|
| Lesion Segmentation      | GAN        | Dermoscopic    | DCGAN            | [Paper](https://example.com)  | [Code ](https://github.com/example) | Fidelity (IoU), Diversity (IS), Clinical Review |

---

### CT Images

| **Application**         | **Type**   | **Modality**   | **Technology**  | **Paper**                           | **Code**                                | **Evaluation**                |
|--------------------------|------------|----------------|------------------|-------------------------------------|-----------------------------------------|-------------------------------|
| Lesion Segmentation      | GAN        | Dermoscopic    | DCGAN            | [Paper](https://example.com)  | [Code ](https://github.com/example) | Fidelity (IoU), Diversity (IS), Clinical Review |

---

### X-ray Images

| **Application**         | **Type**   | **Modality**   | **Technology**  | **Paper**                           | **Code**                                | **Evaluation**                |
|--------------------------|------------|----------------|------------------|-------------------------------------|-----------------------------------------|-------------------------------|
| Lesion Segmentation      | GAN        | Dermoscopic    | DCGAN            | [Paper](https://example.com)  | [Code ](https://github.com/example) | Fidelity (IoU), Diversity (IS), Clinical Review |

---

### OCT Images

| **Application**         | **Type**   | **Modality**   | **Technology**  | **Paper**                           | **Code**                                | **Evaluation**                |
|--------------------------|------------|----------------|------------------|-------------------------------------|-----------------------------------------|-------------------------------|
| Lesion Segmentation      | GAN        | Dermoscopic    | DCGAN            | [Paper](https://example.com)  | [Code ](https://github.com/example) | Fidelity (IoU), Diversity (IS), Clinical Review |

---

### Multiple modalities Images

| **Application**         | **Type**   | **Modality**   | **Technology**  | **Paper**                           | **Code**                                | **Evaluation**                |
|--------------------------|------------|----------------|------------------|-------------------------------------|-----------------------------------------|-------------------------------|
| Lesion Segmentation      | GAN        | Dermoscopic    | DCGAN            | [Paper](https://example.com)  | [Code ](https://github.com/example) | Fidelity (IoU), Diversity (IS), Clinical Review |

---

## **Text**

### Clinical Notes and Radiology Reports

| **Application**          | **Type**       | **Technology**         | **Paper**                           | **Code**                                | **Evaluation**                |
|---------------------------|----------------|-------------------------|-------------------------------------|-----------------------------------------|-------------------------------|
| Clinical Note Gen.        | LLM           | GPT-3                  | [Paper](https://example.com)  | [Code ](https://github.com/example) | Language Coherence (BLEU), Privacy (k-Anonymity) |

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
