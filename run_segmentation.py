import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to sys.path to import ml_core if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except Exception:
    from ml_core import StandardScaler, KMeans, silhouette_score

def run_segmentation():
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "ecommerce_transactions.csv")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("=" * 85)
    print("       OASIS INFOBYTE: DATA ANALYTICS INTERNSHIP (LEVEL 1 - TASK 2)       ")
    print("                 CUSTOMER SEGMENTATION ANALYSIS (RFM + K-MEANS)           ")
    print("=" * 85)
    
    # 1. Load & Inspect
    df = pd.read_csv(data_path)
    print(f"\n--- 1. DATASET STRUCTURE ---")
    print(f"Raw Records: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Missing Customer IDs: {df['Customer_ID'].isnull().sum()}")
    
    # Clean missing Customer IDs
    df = df.dropna(subset=["Customer_ID"]).copy()
    df["Invoice_Date"] = pd.to_datetime(df["Invoice_Date"])
    print(f"Cleaned Records: {len(df)} across {df['Customer_ID'].nunique()} unique customers.")
    
    # 2. Descriptive Customer Statistics
    avg_purchase_val = df["Total_Spend"].mean()
    cust_totals = df.groupby("Customer_ID")["Total_Spend"].sum()
    cust_orders = df.groupby("Customer_ID")["Invoice_No"].nunique()
    
    print(f"\n--- 2. DESCRIPTIVE CUSTOMER METRICS ---")
    print(f"Average Transaction Value (AOV): ${avg_purchase_val:.2f}")
    print(f"Average Purchase Frequency: {cust_orders.mean():.2f} orders per customer")
    print(f"Average Customer Lifetime Value (CLV Spend): ${cust_totals.mean():.2f}")
    print(f"Max Customer Lifetime Value: ${cust_totals.max():.2f}")
    
    # 3. Calculate RFM Metrics
    snapshot_date = df["Invoice_Date"].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby("Customer_ID").agg({
        "Invoice_Date": lambda x: (snapshot_date - x.max()).days,
        "Invoice_No": "nunique",
        "Total_Spend": "sum"
    }).reset_index()
    
    rfm.rename(columns={
        "Invoice_Date": "Recency",
        "Invoice_No": "Frequency",
        "Total_Spend": "Monetary"
    }, inplace=True)
    
    print("\n--- 3. RFM TABLE SUMMARY (FIRST 5 CUSTOMERS) ---")
    print(rfm.head().to_string(index=False))
    
    # 4. Data Transformation & Scaling (Log transform to reduce skewness + StandardScaler)
    rfm_log = np.log1p(rfm[["Recency", "Frequency", "Monetary"]].values)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    # 5. Determine Optimal K (Elbow Method & Silhouette Scores)
    inertias = []
    sil_scores = []
    k_range = range(2, 9)
    
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans_temp.fit_predict(rfm_scaled)
        inertias.append(kmeans_temp.inertia_)
        sil_scores.append(silhouette_score(rfm_scaled, labels))
        
    # Plot Elbow & Silhouette
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(list(k_range), inertias, marker='o', color='#2b5c8f', linewidth=2.5)
    axes[0].set_title("Elbow Method for Optimal K (Inertia)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (K)", fontsize=11)
    axes[0].set_ylabel("Within-Cluster Sum of Squares (Inertia)", fontsize=11)
    
    axes[1].plot(list(k_range), sil_scores, marker='s', color='#27ae60', linewidth=2.5)
    axes[1].set_title("Silhouette Score Evaluation across K", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (K)", fontsize=11)
    axes[1].set_ylabel("Silhouette Score", fontsize=11)
    
    plt.tight_layout()
    plot_path1 = os.path.join(plots_dir, "elbow_and_silhouette_analysis.png")
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f"\n[Saved Plot]: {plot_path1}")
    
    # 6. Apply Optimal K-Means (K=4)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
    
    # Profile Clusters and Assign Meaningful Segment Names
    cluster_means = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().reset_index()
    sorted_clusters = cluster_means.sort_values(by="Monetary", ascending=False)["Cluster"].tolist()
    persona_map = {
        sorted_clusters[0]: "Champions / High-Value VIPs",
        sorted_clusters[1]: "Loyal Core Customers",
        sorted_clusters[2]: "Recent / Potential Loyalists",
        sorted_clusters[3]: "At-Risk / Hibernating"
    }
    rfm["Segment_Name"] = rfm["Cluster"].map(persona_map)
    
    print("\n--- 4. CLUSTER PROFILE SUMMARY (AVERAGES) ---")
    summary = rfm.groupby("Segment_Name").agg(
        Customer_Count=("Customer_ID", "count"),
        Avg_Recency_Days=("Recency", "mean"),
        Avg_Frequency_Orders=("Frequency", "mean"),
        Avg_Monetary_Spend=("Monetary", "mean"),
        Total_Segment_Revenue=("Monetary", "sum")
    ).reset_index()
    print(summary.to_string(index=False))
    
    # 7. Visualizations: 2D Scatter Plots & Segment Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    palette = {"Champions / High-Value VIPs": "#27ae60", "Loyal Core Customers": "#2980b9", 
               "Recent / Potential Loyalists": "#f39c12", "At-Risk / Hibernating": "#e74c3c"}
    
    # Scatter 1: Recency vs Monetary
    sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Segment_Name", palette=palette, alpha=0.8, s=60, ax=axes[0])
    axes[0].set_title("Customer Clusters: Recency vs. Monetary Value", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Recency (Days since last purchase)", fontsize=11)
    axes[0].set_ylabel("Total Monetary Spend ($)", fontsize=11)
    axes[0].legend(loc="upper right", frameon=True)
    
    # Scatter 2: Frequency vs Monetary
    sns.scatterplot(data=rfm, x="Frequency", y="Monetary", hue="Segment_Name", palette=palette, alpha=0.8, s=60, ax=axes[1])
    axes[1].set_title("Customer Clusters: Frequency vs. Monetary Value", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Frequency (Total Invoices)", fontsize=11)
    axes[1].set_ylabel("Total Monetary Spend ($)", fontsize=11)
    axes[1].legend(loc="upper left", frameon=True)
    
    plt.tight_layout()
    plot_path2 = os.path.join(plots_dir, "customer_segments_2d_scatter.png")
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"[Saved Plot]: {plot_path2}")
    
    # Bar Chart: Customer Distribution per Segment
    plt.figure(figsize=(10, 6))
    seg_counts = rfm["Segment_Name"].value_counts().reset_index()
    seg_counts.columns = ["Segment_Name", "Count"]
    sns.barplot(data=seg_counts, x="Segment_Name", y="Count", hue="Segment_Name", palette=palette, legend=False)
    plt.title("Customer Distribution Across Identified Behavioral Segments", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Customer Segment Persona", fontsize=11)
    plt.ylabel("Number of Customers", fontsize=11)
    plt.xticks(rotation=15)
    for p in plt.gca().patches:
        plt.gca().annotate(f"{int(p.get_height())} ({p.get_height()/len(rfm)*100:.1f}%)", 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plot_path3 = os.path.join(plots_dir, "segment_distribution_barchart.png")
    plt.savefig(plot_path3, dpi=300)
    plt.close()
    print(f"[Saved Plot]: {plot_path3}")
    
    print("\n--- 5. TARGETED MARKETING ACTION PLAYBOOK ---")
    print("1. Champions / VIPs (High Spend, High Frequency, Low Recency):")
    print("   -> Action: Early access to new product launches, dedicated VIP concierge support, exclusive loyalty tiered rewards. Avoid aggressive discounts.")
    print("2. Loyal Core Customers (Consistent Frequency & Good Spend):")
    print("   -> Action: Cross-sell complementary categories, offer milestone anniversary gifts, and encourage subscription-based repeat orders.")
    print("3. Recent / Potential Loyalists (Low Recency, Emerging Spend):")
    print("   -> Action: Onboarding email sequences, personalized product recommendations based on first purchase, and second-order discount coupon.")
    print("4. At-Risk / Hibernating (High Recency, Low Frequency):")
    print("   -> Action: Automated 'We Miss You' win-back campaigns, feedback surveys to uncover churn reasons, and limited-time reactivation incentives.")
    print("=" * 85)
    print("Customer Segmentation Completed Successfully!")

if __name__ == "__main__":
    run_segmentation()
