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

### Table 1: Electronic Health Records

| **Application**            | **Type**   | **Technology**      | **Paper**                               | **Code**                                | **Evaluation**            |
|-----------------------------|------------|----------------------|-----------------------------------------|-----------------------------------------|---------------------------|
| Patient Demographics Gen.  | GAN        | Tabular GAN          | [Paper Link](https://example.com)      | [Code Link](https://github.com/example) | Fidelity (MSE)            |
| Disease Progression Model  | VAE        | Bayesian VAE         | [Paper Link](https://example.com)      | [Code Link](https://github.com/example) | Fidelity (KL Divergence), Privacy (k-Anonymity) |

---

## **Signals**

### Table 2: Electrocardiogram (ECG)

| **Application**         | **Type**   | **Modality**   | **Technology**  | **Paper**                           | **Code**                                | **Evaluation**                |
|--------------------------|------------|----------------|------------------|-------------------------------------|-----------------------------------------|-------------------------------|
| ECG Generation           | GAN        | 12-lead ECG    | LSTM-GAN         | [Paper Link](https://example.com)  | [Code Link](https://github.com/example) | Fidelity (MSE), Clinical Review |
| PPG-to-ECG Translation   | DM         | PPG to ECG     | DDPM             | [Paper Link](https://example.com)  | [Code Link](https://github.com/example) | Fidelity (PSNR), Diversity (FID) |

### Table 3: Other Signals

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
