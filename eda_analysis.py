"""
==============================================================================
Project: Retail Sales Analysis
Module: eda_analysis.py
Description: Performs in-depth Exploratory Data Analysis (EDA) on the cleaned
             retail dataset. Computes core retail KPIs, monthly trend analysis,
             category/regional breakdowns, and product performance analysis.
==============================================================================
"""

import os
import pandas as pd
import numpy as np


def load_cleaned_data(filepath: str = "data/processed/retail_sales_cleaned.csv") -> pd.DataFrame:
    """Loads the preprocessed retail dataset with proper datetimes."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cleaned dataset not found at {filepath}. Please run data_cleaning.py first.")
    df = pd.read_csv(filepath)
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    return df


def calculate_executive_kpis(df: pd.DataFrame) -> dict:
    """
    Computes top-level retail Key Performance Indicators (KPIs).
    """
    total_sales = df["Net_Sales"].sum()
    gross_sales = df["Gross_Sales"].sum()
    total_profit = df["Profit"].sum()
    total_cost = df["Total_Cost"].sum()
    total_orders = df["Transaction_ID"].nunique()
    total_units_sold = df["Quantity"].sum()
    aov = total_sales / total_orders if total_orders > 0 else 0.0
    profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0.0
    avg_discount = df["Discount"].mean() * 100

    kpis = {
        "Total_Net_Sales": total_sales,
        "Total_Gross_Sales": gross_sales,
        "Total_Profit": total_profit,
        "Total_Cost": total_cost,
        "Total_Orders": total_orders,
        "Total_Units_Sold": total_units_sold,
        "Average_Order_Value": aov,
        "Overall_Profit_Margin_Pct": profit_margin,
        "Average_Discount_Pct": avg_discount
    }
    return kpis


def analyze_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates sales, profit, orders, and MoM growth rate by month.
    """
    monthly = (
        df.groupby(["Year", "Month", "Month_Name", "Year_Month"])
        .agg(
            Net_Sales=("Net_Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Transaction_ID", "count"),
            Units_Sold=("Quantity", "sum")
        )
        .reset_index()
        .sort_values(by=["Year", "Month"])
    )

    # Calculate MoM (Month-over-Month) Growth
    monthly["MoM_Sales_Growth_Pct"] = monthly["Net_Sales"].pct_change() * 100
    monthly["Profit_Margin_Pct"] = (monthly["Profit"] / monthly["Net_Sales"]) * 100
    monthly["AOV"] = monthly["Net_Sales"] / monthly["Orders"]

    return monthly


def analyze_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes sales, profit, margin, and order volume by Product Category.
    """
    category_perf = (
        df.groupby("Product_Category")
        .agg(
            Net_Sales=("Net_Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Transaction_ID", "count"),
            Units_Sold=("Quantity", "sum"),
            Avg_Discount=("Discount", lambda x: x.mean() * 100)
        )
        .reset_index()
    )
    category_perf["Profit_Margin_Pct"] = (category_perf["Profit"] / category_perf["Net_Sales"]) * 100
    category_perf["Revenue_Share_Pct"] = (category_perf["Net_Sales"] / df["Net_Sales"].sum()) * 100
    category_perf = category_perf.sort_values(by="Net_Sales", ascending=False).reset_index(drop=True)
    return category_perf


def analyze_regions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes geographic performance across regions.
    """
    region_perf = (
        df.groupby("Region")
        .agg(
            Net_Sales=("Net_Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Transaction_ID", "count"),
            Avg_AOV=("Net_Sales", "mean")
        )
        .reset_index()
    )
    region_perf["Profit_Margin_Pct"] = (region_perf["Profit"] / region_perf["Net_Sales"]) * 100
    region_perf["Sales_Share_Pct"] = (region_perf["Net_Sales"] / df["Net_Sales"].sum()) * 100
    region_perf = region_perf.sort_values(by="Net_Sales", ascending=False).reset_index(drop=True)
    return region_perf


def analyze_products(df: pd.DataFrame, top_n: int = 10):
    """
    Identifies top-performing products by sales and profit, and underperforming products.
    """
    prod_summary = (
        df.groupby(["Product_Category", "Product_Subcategory", "Product_Name"])
        .agg(
            Net_Sales=("Net_Sales", "sum"),
            Profit=("Profit", "sum"),
            Units_Sold=("Quantity", "sum"),
            Orders=("Transaction_ID", "count"),
            Avg_Discount=("Discount", lambda x: x.mean() * 100)
        )
        .reset_index()
    )
    prod_summary["Profit_Margin_Pct"] = (prod_summary["Profit"] / prod_summary["Net_Sales"]) * 100

    top_revenue = prod_summary.sort_values(by="Net_Sales", ascending=False).head(top_n).reset_index(drop=True)
    top_profit = prod_summary.sort_values(by="Profit", ascending=False).head(top_n).reset_index(drop=True)
    bottom_performers = prod_summary.sort_values(by="Profit", ascending=True).head(5).reset_index(drop=True)

    return top_revenue, top_profit, bottom_performers


def run_full_eda(filepath: str = "data/processed/retail_sales_cleaned.csv"):
    """
    Executes full EDA workflow and prints clean summary reports.
    """
    df = load_cleaned_data(filepath)

    print("=" * 70)
    print("              RETAIL SALES ANALYSIS - EXECUTIVE REPORT")
    print("=" * 70)

    # 1. KPIs
    kpis = calculate_executive_kpis(df)
    print("\n>>> 1. KEY PERFORMANCE INDICATORS (KPIs)")
    print("-" * 50)
    print(f"Total Net Sales Revenue  : ${kpis['Total_Net_Sales']:,.2f}")
    print(f"Total Gross Sales        : ${kpis['Total_Gross_Sales']:,.2f}")
    print(f"Total Cost of Goods Sold : ${kpis['Total_Cost']:,.2f}")
    print(f"Total Net Profit         : ${kpis['Total_Profit']:,.2f}")
    print(f"Overall Profit Margin    : {kpis['Overall_Profit_Margin_Pct']:.2f}%")
    print(f"Total Completed Orders   : {kpis['Total_Orders']:,}")
    print(f"Total Units Sold         : {kpis['Total_Units_Sold']:,}")
    print(f"Average Order Value (AOV): ${kpis['Average_Order_Value']:,.2f}")
    print(f"Average Discount Rate    : {kpis['Average_Discount_Pct']:.2f}%")

    # 2. Monthly Trends
    monthly = analyze_monthly_trends(df)
    print("\n>>> 2. MONTHLY SALES & PROFIT PERFORMANCE")
    print("-" * 75)
    print(monthly[["Year_Month", "Net_Sales", "Profit", "Profit_Margin_Pct", "Orders", "MoM_Sales_Growth_Pct"]].to_string(
        index=False,
        formatters={
            "Net_Sales": "${:,.2f}".format,
            "Profit": "${:,.2f}".format,
            "Profit_Margin_Pct": "{:.1f}%".format,
            "Orders": "{:,}".format,
            "MoM_Sales_Growth_Pct": lambda x: f"{x:+.1f}%" if pd.notnull(x) else "N/A"
        }
    ))

    # 3. Category Breakdown
    categories = analyze_categories(df)
    print("\n>>> 3. PRODUCT CATEGORY BREAKDOWN")
    print("-" * 75)
    print(categories[["Product_Category", "Net_Sales", "Revenue_Share_Pct", "Profit", "Profit_Margin_Pct", "Orders"]].to_string(
        index=False,
        formatters={
            "Net_Sales": "${:,.2f}".format,
            "Revenue_Share_Pct": "{:.1f}%".format,
            "Profit": "${:,.2f}".format,
            "Profit_Margin_Pct": "{:.1f}%".format,
            "Orders": "{:,}".format
        }
    ))

    # 4. Regional Breakdown
    regions = analyze_regions(df)
    print("\n>>> 4. REGIONAL PERFORMANCE")
    print("-" * 75)
    print(regions[["Region", "Net_Sales", "Sales_Share_Pct", "Profit", "Profit_Margin_Pct", "Orders", "Avg_AOV"]].to_string(
        index=False,
        formatters={
            "Net_Sales": "${:,.2f}".format,
            "Sales_Share_Pct": "{:.1f}%".format,
            "Profit": "${:,.2f}".format,
            "Profit_Margin_Pct": "{:.1f}%".format,
            "Orders": "{:,}".format,
            "Avg_AOV": "${:,.2f}".format
        }
    ))

    # 5. Top & Bottom Products
    top_rev, top_prof, bottom_prof = analyze_products(df)
    print("\n>>> 5. TOP 5 REVENUE GENERATING PRODUCTS")
    print("-" * 75)
    print(top_rev.head(5)[["Product_Name", "Product_Category", "Net_Sales", "Profit", "Profit_Margin_Pct", "Units_Sold"]].to_string(
        index=False,
        formatters={
            "Net_Sales": "${:,.2f}".format,
            "Profit": "${:,.2f}".format,
            "Profit_Margin_Pct": "{:.1f}%".format,
            "Units_Sold": "{:,}".format
        }
    ))

    print("\n>>> 6. BOTTOM 5 UNDERPERFORMING PRODUCTS (BY PROFIT)")
    print("-" * 75)
    print(bottom_prof[["Product_Name", "Product_Category", "Net_Sales", "Profit", "Profit_Margin_Pct", "Avg_Discount"]].to_string(
        index=False,
        formatters={
            "Net_Sales": "${:,.2f}".format,
            "Profit": "${:,.2f}".format,
            "Profit_Margin_Pct": "{:.1f}%".format,
            "Avg_Discount": "{:.1f}%".format
        }
    ))
    print("=" * 70)


if __name__ == "__main__":
    run_full_eda()
