# Intelligent Customer Churn Prediction & Agentic Retention Assistant

**Rudraksh Rathod (Enrollment No: 2401010396)**  
**Kaustubh Hiwanj (Enrollment No: 2401010217)**  
**Bhargav Patil (Enrollment No: 2401020092)**  
**April 19, 2026**

---

## Abstract

This project presents a state-of-the-art customer retention system that integrates classical machine learning with autonomous agentic reasoning. Building upon a Milestone 1 baseline using a Random Forest Classifier to predict churn probability, the system evolves into an **Agentic AI Retention Assistant** in Milestone 2. By leveraging a **Retrieval-Augmented Generation (RAG)** pipeline powered by **ChromaDB** and the **Gemini-2.5-Flash-Lite** model, the system autonomously retrieves industry-standard retention strategies and synthesizes them into personalized, actionable guidance for high-risk customers. This approach bridges the gap between predictive analytics and strategic decision-making in the telecommunications sector.

---

## 1. Introduction & Problem Statement

In the highly competitive telecommunications industry, customer churn—the loss of subscribers to competitors—is a primary driver of revenue decline. Traditional churn management often relies on reactive measures or static reports that provide little room for personalized intervention. 

While **Milestone 1** successfully established a robust ML pipeline for identifying high-risk customers, **Milestone 2** addresses the critical need for a system that doesn't just predict churn but actively assists in preventing it. The core challenge lies in translating raw churn probabilities and complex customer demographics into consistent, effective, and professional retention actions. This project implements an autonomous agent capable of reasoning over customer data and a curated knowledge base of retention strategies to provide 24/7 decision support.

---

## 2. Detailed Data Description

The system utilizes a hybrid data architecture to power both its predictive and agentic components:

### 2.1 Structured Customer Data (ML Pipeline)
- **Dataset:** Telco Customer Churn Dataset (~7,000 records).
- **Features:** 21 features including demographics (gender, senior citizen status), service details (Internet service, Streaming TV, Multiple lines), and account information (tenure, contract type, payment method, monthly charges).
- **Pre-processing:** Categorical variables were encoded using Label Encoding, and non-predictive features like `customerID` were removed to ensure model generalization.

### 2.2 Unstructured Retention Strategies (RAG Pipeline)
- **Source:** `strategies.pdf` (Industry best practices and company-specific retention policies).
- **Processing:** To ground the Agentic AI in verified strategies, we implemented a RAG pipeline. Clinical/Business PDFs were ingested and split into chunks of **1000 characters** with a **200-character overlap** to maintain semantic continuity. These chunks were indexed in a **ChromaDB** vector database using **Gemini-001** embeddings.

---

## 3. Exploratory Data Analysis (EDA) Processes

EDA was instrumental in uncovering the underlying patterns that drive churn:
- **Feature Importance:** Analysis revealed that **Contract Type (Month-to-month)**, **Tenure**, and **Total Charges** are the strongest predictors of churn.
- **Service Correlations:** Customers with Fiber Optic internet and those without Tech Support showed significantly higher churn rates.
- **Data Integrity:** Confirmed zero missing values and handled class imbalance (~27% churn rate) through optimized model weighting.

---

## 4. Methodology

The system architecture follows a cascaded design, progressing from statistical prediction to agentic reasoning.

### 4.1 ML Pipeline: Random Forest Classifier
We utilized a **Random Forest Classifier** for the baseline prediction. This ensemble method was chosen for its robustness against noise and ability to handle non-linear interactions between customer features.

### 4.2 Agentic Workflow (LangChain)
The Milestone 2 extension implements a sophisticated RAG-driven agent:
1. **Context Ingestion:** The agent receives the customer's profile (tenure, services, contract) and the churn probability predicted by the ML model.
2. **Retrieval (ChromaDB):** Using the customer's specific pain points (e.g., high monthly charges or lack of tech support), the system performs a similarity search in ChromaDB.
3. **Reasoning & Generation:** The retrieved strategy chunks are passed to the **Gemini-2.5-Flash-Lite** model, which synthesizes a personalized retention plan.

### 4.3 Conversational Interface & Guardrails
The system features an interactive Streamlit dashboard:
- **Context-Aware Chat:** Users can ask follow-up questions like "What specific offer can we give to reduce their monthly bill?" or "How do I explain the benefits of a 2-year contract?"
- **Strict Guardrails:** The agent is prompted to act strictly as a **Telecom Retention Strategist**. It includes mandatory disclaimers and refuses to answer non-business related queries to maintain professional focus.

---

## 5. Mathematical Formulation

### 5.1 Stage 1: Random Forest (Ensemble Learning)
The final prediction $H(x)$ is an aggregation of $T$ individual decision trees $h_t(x)$:
$$H(x) = \text{argmax}_y \sum_{t=1}^{T} I(h_t(x) = y)$$
This reduces variance and prevents the overfitting common in single decision trees.

### 5.2 Stage 2: RAG Retrieval (Cosine Similarity)
Retrieval within ChromaDB is performed by calculating the **Cosine Similarity** between the query embedding ($\vec{q}$) and the document chunk embeddings ($\vec{d}$):
$$\text{similarity} = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}$$
We utilize the **gemini-embedding-001** model for generating high-dimensional dense vector embeddings.

---

## 6. Optimization Strategies

- **Hyperparameter Tuning:** The Random Forest was optimized using **GridSearchCV**, settling on `n_estimators=200` and `max_depth=11` to balance accuracy and generalization.
- **Hallucination Mitigation:** We implemented a "Knowledge-Grounded" prompt template that forces the LLM to cite provided strategies, ensuring that retention offers are realistic and company-approved.
- **Efficient Embedding:** The use of recursive character splitting ensures that chunks are semantically coherent, improving the precision of the RAG retrieval.

---

## 7. Evaluation & Results

### 7.1 ML Pipeline Performance
The Random Forest model achieved strong results, proving a reliable baseline for the agent.

| Metric | Score |
| :--- | :--- |
| **Testing Accuracy** | 80.7% |
| **Overall Accuracy (Post-Tuning)** | 89.0% |
| **Weighted F1-Score** | 0.80 |
| **Precision (Churn)** | 0.68 |

### 7.2 Agentic AI Qualitative Analysis
The Agentic Assistant successfully generates:
- **Personalized Risk Summaries:** Identifying why a specific customer is likely to churn.
- **Actionable Offers:** Recommending specific discounts or service upgrades grounded in the `strategies.pdf` knowledge base.
- **Natural Language Explanations:** Providing clear reasoning for the suggested strategy.

---

## 8. References

1. **Telco Customer Churn Dataset:** [Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. **Google Gemini API:** [Gemini Model Documentation](https://ai.google.dev/models/gemini)
3. **ChromaDB:** [Vector Database Docs](https://docs.trychroma.com/)

---

## 9. Team Contribution

- **Rudraksh Rathod:** Project Lead; Designed the end-to-end architecture and led the ML Preprocessing and EDA phases.
- **Kaustubh Hiwanj:** Implemented the RAG pipeline and ChromaDB integration; Developed the LangChain agentic workflow.
- **Bhargav Patil:** Developed the Random Forest model, performed hyperparameter tuning, and integrated the Streamlit UI components.
