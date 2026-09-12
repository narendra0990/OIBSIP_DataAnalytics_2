# 🛍 Customer Segmentation Analysis using RFM & K-Means Clustering

**Organization:** Oasis Infobyte (OIBSIP)  
**Track:** Data Analytics (Level 1 • Task 2)  
**Author:** Narendra

---

## 📌 Project Overview
This project implements an unsupervised Machine Learning pipeline to segment e-commerce customers based on the **RFM (Recency, Frequency, Monetary)** behavioral framework. Using the Elbow Method and Silhouette Analysis, the model identifies 4 actionable commercial personas and provides tailored retention strategies.

---

## 📁 Repository Structure
```text
├── data/
│   └── ecommerce_transactions.csv
├── notebooks/
│   └── customer_segmentation.ipynb
├── charts/
│   ├── elbow_and_silhouette_analysis.png
│   ├── customer_segments_2d_scatter.png
│   └── segment_distribution_barchart.png
├── ml_core.py
├── run_segmentation.py
├── requirements.txt
└── README.md
```

---

## 👥 Customer Personas Identified
- **Champions / High-Value VIPs**: Highest spend, highest order frequency, lowest recency.
- **Loyal Core Customers**: Consistent transaction frequency with steady spend.
- **Recent / Potential Loyalists**: New customers with low recency and growing order activity.
- **At-Risk / Hibernating**: High dormancy with low historical frequency requiring win-back campaigns.

---

## 🚀 How to Run Locally
```bash
pip install -r requirements.txt
python run_segmentation.py
jupyter notebook notebooks/customer_segmentation.ipynb
```
