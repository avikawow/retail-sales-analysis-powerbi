"""
==============================================================================
Project: Retail Sales Analysis
Module: generate_visualizations.py
Description: Generates publication-ready visualizations using Matplotlib & Seaborn
             and saves them into the 'reports/figures/' directory.
==============================================================================
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np


# Set global aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial", "DejaVu Sans", "Helvetica"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def ensure_output_dir(dir_path: str = "reports/figures"):
    """Creates the figures directory if it doesn't already exist."""
    os.makedirs(dir_path, exist_ok=True)


def plot_monthly_sales_trend(df: pd.DataFrame, output_dir: str = "reports/figures"):
    """
    Generates a dual-axis monthly sales and profit trend chart.
    """
    monthly = (
        df.groupby(["Year", "Month", "Month_Name", "Year_Month"])
        .agg(
            Net_Sales=("Net_Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Transaction_ID", "count")
        )
        .reset_index()
        .sort_values(by=["Year", "Month"])
    )

    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)

    # Primary Axis: Net Sales
    color_sales = "#1f77b4"
    line1 = ax1.plot(
        monthly["Month_Name"],
        monthly["Net_Sales"] / 1000,
        marker="o",
        linewidth=2.8,
        color=color_sales,
        label="Net Sales ($K)"
    )
    ax1.fill_between(monthly["Month_Name"], monthly["Net_Sales"] / 1000, color=color_sales, alpha=0.12)
    ax1.set_ylabel("Net Sales ($ in Thousands)", color=color_sales, fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_sales)
    ax1.set_ylim(40, 110)

    # Data labels on Sales
    for i, val in enumerate(monthly["Net_Sales"] / 1000):
        ax1.annotate(
            f"${val:.1f}K",
            (i, val + 1.8),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=color_sales
        )

    # Secondary Axis: Net Profit
    ax2 = ax1.twinx()
    color_profit = "#2ca02c"
    line2 = ax2.plot(
        monthly["Month_Name"],
        monthly["Profit"] / 1000,
        marker="s",
        linewidth=2.5,
        linestyle="--",
        color=color_profit,
        label="Net Profit ($K)"
    )
    ax2.set_ylabel("Net Profit ($ in Thousands)", color=color_profit, fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_profit)
    ax2.set_ylim(8, 28)

    # Data labels on Profit
    for i, val in enumerate(monthly["Profit"] / 1000):
        ax2.annotate(
            f"${val:.1f}K",
            (i, val - 1.6),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=color_profit
        )

    # Title & Legends
    plt.title("2023 Monthly Sales & Profit Performance Trend", fontsize=15, fontweight="bold", pad=20)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    ax1.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_file = os.path.join(output_dir, "monthly_sales_trend.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"[CHART SAVED] {out_file}")


def plot_category_performance(df: pd.DataFrame, output_dir: str = "reports/figures"):
    """
    Generates a horizontal comparison bar chart of Sales and Profit by Category.
    """
    cat_df = (
        df.groupby("Product_Category")
        .agg(
            Net_Sales=("Net_Sales", "sum"),
            Profit=("Profit", "sum")
        )
        .reset_index()
    )
    cat_df["Profit_Margin_Pct"] = (cat_df["Profit"] / cat_df["Net_Sales"]) * 100
    cat_df = cat_df.sort_values(by="Net_Sales", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    y_pos = np.arange(len(cat_df))
    bar_height = 0.38

    # Sales & Profit bars
    rects1 = ax.barh(y_pos + bar_height/2, cat_df["Net_Sales"] / 1000, bar_height, label="Net Sales ($K)", color="#2b5c8f")
    rects2 = ax.barh(y_pos - bar_height/2, cat_df["Profit"] / 1000, bar_height, label="Net Profit ($K)", color="#46a066")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat_df["Product_Category"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Amount ($ in Thousands)", fontsize=11, fontweight="bold")
    ax.set_title("Product Category Breakdown: Net Sales vs. Profit", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True)

    # Annotate margin percentage
    for i, (_, row) in enumerate(cat_df.iterrows()):
        sales_k = row["Net_Sales"] / 1000
        margin = row["Profit_Margin_Pct"]
        ax.text(
            sales_k + 8,
            i,
            f"Margin: {margin:.1f}%",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#333333"
        )

    ax.set_xlim(0, cat_df["Net_Sales"].max() / 1000 * 1.22)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_file = os.path.join(output_dir, "category_sales_profit.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"[CHART SAVED] {out_file}")


def plot_regional_performance(df: pd.DataFrame, output_dir: str = "reports/figures"):
    """
    Generates a 3-panel regional analysis: Sales Share, Profit Margin %, and AOV.
    """
    reg_df = (
        df.groupby("Region")
        .agg(
            Net_Sales=("Net_Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Transaction_ID", "count")
        )
        .reset_index()
    )
    reg_df["Profit_Margin_Pct"] = (reg_df["Profit"] / reg_df["Net_Sales"]) * 100
    reg_df["AOV"] = reg_df["Net_Sales"] / reg_df["Orders"]
    reg_df = reg_df.sort_values(by="Net_Sales", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    colors = ["#2b5c8f", "#4388b8", "#6baed6", "#9ecae1", "#c6dbef"]

    # Panel 1: Net Sales by Region
    axes[0].bar(reg_df["Region"], reg_df["Net_Sales"] / 1000, color="#1f77b4", edgecolor="#0e4b75", width=0.55)
    axes[0].set_title("Total Net Sales ($K)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Sales ($ in Thousands)")
    for i, v in enumerate(reg_df["Net_Sales"] / 1000):
        axes[0].text(i, v + 2, f"${v:.1f}K", ha="center", fontsize=9, fontweight="bold")
    axes[0].set_ylim(0, reg_df["Net_Sales"].max() / 1000 * 1.15)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # Panel 2: Profit Margin % by Region
    axes[1].bar(reg_df["Region"], reg_df["Profit_Margin_Pct"], color="#2ca02c", edgecolor="#1b631b", width=0.55)
    axes[1].set_title("Profit Margin (%)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Margin Percentage (%)")
    for i, v in enumerate(reg_df["Profit_Margin_Pct"]):
        axes[1].text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_ylim(20, 31)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    # Panel 3: Average Order Value (AOV) by Region
    axes[2].bar(reg_df["Region"], reg_df["AOV"], color="#ff7f0e", edgecolor="#b85600", width=0.55)
    axes[2].set_title("Average Order Value ($)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("AOV ($)")
    for i, v in enumerate(reg_df["AOV"]):
        axes[2].text(i, v + 4, f"${v:.1f}", ha="center", fontsize=9, fontweight="bold")
    axes[2].set_ylim(240, 340)
    axes[2].grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle("Geographic Sales & Efficiency Benchmarking by Region", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()

    out_file = os.path.join(output_dir, "regional_performance.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"[CHART SAVED] {out_file}")


def plot_top_and_bottom_products(df: pd.DataFrame, output_dir: str = "reports/figures"):
    """
    Plots Top 8 Revenue Generators and Bottom Underperforming Products.
    """
    prod = (
        df.groupby(["Product_Name", "Product_Category"])
        .agg(
            Net_Sales=("Net_Sales", "sum"),
            Profit=("Profit", "sum")
        )
        .reset_index()
    )
    prod["Profit_Margin_Pct"] = (prod["Profit"] / prod["Net_Sales"]) * 100

    top_rev = prod.sort_values(by="Net_Sales", ascending=False).head(8)
    bottom_prof = prod.sort_values(by="Profit", ascending=True).head(5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Subplot 1: Top 8 Products by Revenue
    y_pos1 = np.arange(len(top_rev))
    ax1.barh(y_pos1, top_rev["Net_Sales"] / 1000, color="#1f77b4", edgecolor="#104a75", height=0.6)
    ax1.set_yticks(y_pos1)
    ax1.set_yticklabels(top_rev["Product_Name"], fontsize=9.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Net Sales ($ in Thousands)", fontsize=10, fontweight="bold")
    ax1.set_title("Top 8 Products by Total Revenue", fontsize=13, fontweight="bold", pad=12)
    for i, v in enumerate(top_rev["Net_Sales"] / 1000):
        ax1.text(v + 2, i, f"${v:.1f}K", va="center", fontsize=8.5, fontweight="bold")
    ax1.set_xlim(0, top_rev["Net_Sales"].max() / 1000 * 1.15)
    ax1.grid(axis="x", linestyle=":", alpha=0.6)

    # Subplot 2: Bottom 5 Underperforming Products (by Profit)
    y_pos2 = np.arange(len(bottom_prof))
    bar_colors = ["#d62728" if p < 0 else "#f39c12" for p in bottom_prof["Profit"]]
    ax2.barh(y_pos2, bottom_prof["Profit"] / 1000, color=bar_colors, edgecolor="#555555", height=0.6)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(bottom_prof["Product_Name"], fontsize=9.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Net Profit ($ in Thousands)", fontsize=10, fontweight="bold")
    ax2.set_title("Bottom 5 Products by Profit (Highlighting Loss Makers)", fontsize=13, fontweight="bold", pad=12)
    for i, v in enumerate(bottom_prof["Profit"] / 1000):
        pos_offset = 0.1 if v >= 0 else -0.15
        ha = "left" if v >= 0 else "right"
        ax2.text(v + pos_offset, i, f"${v*1000:.0f}", va="center", ha=ha, fontsize=8.5, fontweight="bold")
    ax2.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax2.grid(axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_file = os.path.join(output_dir, "top_bottom_products.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"[CHART SAVED] {out_file}")


def plot_profit_margin_distribution(df: pd.DataFrame, output_dir: str = "reports/figures"):
    """
    Plots the profit margin distribution and boxplot by Category.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot 1: Distribution of Profit Margin Pct
    sns.histplot(
        df["Profit_Margin_Pct"],
        bins=35,
        kde=True,
        color="#2b5c8f",
        ax=ax1,
        edgecolor="white"
    )
    ax1.axvline(df["Profit_Margin_Pct"].median(), color="red", linestyle="--", linewidth=1.8, label=f"Median: {df['Profit_Margin_Pct'].median():.1f}%")
    ax1.set_title("Distribution of Transaction Profit Margins (%)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Profit Margin (%)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Number of Transactions", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Profit Margin Boxplot by Category
    order = df.groupby("Product_Category")["Profit_Margin_Pct"].median().sort_values(ascending=False).index
    sns.boxplot(
        data=df,
        y="Product_Category",
        x="Profit_Margin_Pct",
        order=order,
        hue="Product_Category",
        palette="Blues_r",
        legend=False,
        ax=ax2,
        fliersize=3
    )
    ax2.set_title("Profit Margin Spread across Product Categories", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Profit Margin (%)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("")
    ax2.axvline(0, color="red", linestyle=":", linewidth=1.2)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_file = os.path.join(output_dir, "profit_margin_distribution.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"[CHART SAVED] {out_file}")


def generate_all_visualizations(
    csv_path: str = "data/processed/retail_sales_cleaned.csv",
    output_dir: str = "reports/figures"
):
    """
    Loads cleaned retail data and runs all visualization routines.
    """
    ensure_output_dir(output_dir)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {csv_path}")

    print("=" * 60)
    print(">>> GENERATING ANALYTICS VISUALIZATIONS")
    print("=" * 60)
    df = pd.read_csv(csv_path)

    plot_monthly_sales_trend(df, output_dir)
    plot_category_performance(df, output_dir)
    plot_regional_performance(df, output_dir)
    plot_top_and_bottom_products(df, output_dir)
    plot_profit_margin_distribution(df, output_dir)

    print(f"[SUCCESS] All 5 high-resolution charts generated in '{output_dir}'.\n")


if __name__ == "__main__":
    generate_all_visualizations()
