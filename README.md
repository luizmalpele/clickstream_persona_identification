# clickstream_persona_identification
An analysis of user behavior segmentation and funnel friction on 110M e-commerce clickstream events using Mini-Batch K-Means, LightGBM, and SHAP.

# User Behavior Segmentation and Funnel Friction Modeling in E-Commerce Clickstreams

**Author:** Luiz Gustavo Fagundes Malpele, Pranidhi Prabhat, Rei Sanchez-Arias

## Abstract & Project Overview
This project investigates user session behavior by leveraging funnel page-view data, product metadata, and purchase behavior to understand funnel progression and conversion in a large, multi-category e-commerce environment. The primary objective is to identify distinct user cohorts (personas) and utilize supervised learning techniques to uncover data patterns that indicate friction in the user funnel. These insights provide data-driven suggestions for personalized interventions and targeted A/B tests to improve the overall shopping experience.

## Dataset & Optimization
*   **Data Source:** Multi-category eCommerce dataset derived from Kaggle, originally containing 110 million user events over a two-month period.
*   **Data Pretreatment:** Missing categorical values were intelligently filled by generalizing child categories to preserve parent-category groupings without falling into the false precision trap. After cleaning, duplicate removal, and NaN remediation, the final event-level dataset retained 90,070,647 rows (81.9% of the original data).
*   **Computational Throughput:** Due to the massive scale of the data, the project utilized a High-RAM Virtual Machine via Google Colab. Memory optimization strategies included downcasting data types (e.g., `np.float32`, `np.int8`), prioritizing vectorized operations, and utilizing power-of-two batch sizes ($2^{21}$ and $2^{22}$) to maximize memory bandwidth efficiency during matrix operations.

## Methodology

### 1. Unsupervised Learning & Persona Building
*   **Feature Engineering & Selection:** 44 initial features were pruned through a four-stage multicollinearity reduction process down to 8 core behavioral pillars (e.g., price elasticity index, brand switcher, session intensity).
*   **Clustering:** The Elbow Method was paired with Mini-Batch K-Means to identify $k=8$ optimal user personas.
*   **Dimensionality Reduction:** Principal Component Analysis (PCA) was used to denoise the data, capturing 93% of the behavioral variance across 5 components. This was combined with t-SNE (perplexity = 50) to visually map and validate the spatial stability and cohesive "DNA" of the user segments.

### 2. Supervised Learning & Friction Diagnosis
*   **Modeling:** LightGBM was selected over Logistic Regression for its ability to scale efficiently to 20 million session records while capturing non-linear relationships.
*   **Explainability:** TreeSHAP (SHapley Additive exPlanations) was implemented to assign feature importance and investigate complex dependencies.

## Key Findings
*   **The 8 Personas:** The clustering algorithm successfully segmented users into distinct groups, including Regular Window-Shoppers, High-Intensity Bouncers, Smartphone Searchers, Extreme Researchers, Brand Switchers, Careful Researchers, Price Sensitive Users, and Focused Buyers.
*   **The Session Intensity Threshold:** TreeSHAP analysis identified "session intensity" as the primary negative driver of session conversion. Once a user's session intensity exceeded 7.5 actions per minute, conversion probability sharply declined.
*   **The Behavioral Paradox:** A specific "high-friction" cohort (comprising 3.04% of traffic) exhibited a cart-to-view ratio 3.7x higher than the healthy baseline, indicating strong purchase intent. However, due to interface barriers driving their session intensity to 11x the baseline, their actual conversion rate plummeted by 96% (dropping to 0.28%).

## Strategic Personalization & Next Steps
Based on the distinctive behavioral DNA of the modeled personas, the following targeted interventions are recommended for A/B testing:
*   **Extreme & Careful Researchers:** Implement inline product comparison tools to reduce page "pogo-sticking" and satisfy deep catalog exploration needs while lowering session intensity.
*   **Price-Sensitive Users & Smartphone Searchers:** Trigger dynamic discount banners or bundle offers early in the session when high price elasticity behavior is detected.
*   **Focused Buyers:** Capitalize on this high-converting group (20.16% CVR) by testing a streamlined, one-click checkout flow that bypasses the standard multi-step cart process to minimize friction.

## Repository Contents
*   `exploratory_data_analysis.ipynb`: Complete Python codebase for data pretreatment, feature engineering, Mini-Batch K-Means clustering, PCA/t-SNE dimensionality reduction, LightGBM modeling, and SHAP explainability.
