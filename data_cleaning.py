"""
==============================================================================
Project: Retail Sales Analysis
Module: data_cleaning.py
Description: Performs comprehensive data cleaning, anomaly detection, type 
             conversion, missing value imputation, duplicate removal, and 
             feature engineering. Produces a pristine, Power BI-ready dataset.
==============================================================================
"""

import os
import pandas as pd
import numpy as np


def clean_retail_data(
    input_path: str = "data/raw/retail_sales_raw.csv",
    output_path: str = "data/processed/retail_sales_cleaned.csv"
) -> pd.DataFrame:
    """
    Cleans raw retail transaction records and exports a standardized dataset.

    Parameters:
    -----------
    input_path : str
        Filepath to the raw CSV dataset.
    output_path : str
        Filepath to save the cleaned, Power BI-ready CSV.

    Returns:
    --------
    pd.DataFrame: The cleaned and enriched DataFrame.
    """
    print("=" * 60)
    print(">>> STARTING DATA CLEANING PIPELINE")
    print("=" * 60)

    # 1. Load Raw Dataset
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source file not found at: {input_path}")

    df_raw = pd.read_csv(input_path)
    initial_rows, initial_cols = df_raw.shape
    print(f"[STEP 1] Loaded raw data: {initial_rows:,} rows, {initial_cols} columns.")

    # 2. Duplicate Detection and Removal
    duplicates_count = df_raw.duplicated().sum()
    print(f"[STEP 2] Detected {duplicates_count} exact duplicate rows.")
    df = df_raw.drop_duplicates().copy()

    # Also ensure Transaction_ID uniqueness if duplicates exist with same ID
    txn_dup_count = df.duplicated(subset=["Transaction_ID"]).sum()
    if txn_dup_count > 0:
        print(f"         Detected {txn_dup_count} duplicate Transaction_IDs. Keeping first occurrence.")
        df = df.drop_duplicates(subset=["Transaction_ID"], keep="first").copy()

    # 3. Handle String Cleaning & Formatting
    print("[STEP 3] Cleaning text fields (stripping whitespace, normalizing casing)...")
    text_columns = [
        "Customer_Segment", "Region", "Product_Category", 
        "Product_Subcategory", "Product_Name", "Payment_Method", "Shipping_Status"
    ]
    for col in text_columns:
        if col in df.columns:
            # Strip whitespace and normalize title casing
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan})

    # Specific normalization for Region
    if "Region" in df.columns:
        df["Region"] = df["Region"].str.title()

    # Specific normalization for Product_Category
    if "Product_Category" in df.columns:
        df["Product_Category"] = df["Product_Category"].str.strip()

    # 4. Handle Data Types & Numeric Columns
    print("[STEP 4] Standardizing numeric types and cleaning currency symbols...")
    # Clean Unit_Price: remove '$' and commas if present, convert to float
    if "Unit_Price" in df.columns:
        df["Unit_Price"] = (
            df["Unit_Price"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["Unit_Price"] = pd.to_numeric(df["Unit_Price"], errors="coerce")

    # Numeric conversion for quantities and costs
    df["Unit_Cost"] = pd.to_numeric(df["Unit_Cost"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce").fillna(0.0)

    # Filter out invalid quantities (e.g. quantity <= 0)
    invalid_qty_count = (df["Quantity"] <= 0).sum()
    print(f"         Found {invalid_qty_count} records with non-positive Quantity. Filtering out.")
    df = df[df["Quantity"] > 0].copy()
    df["Quantity"] = df["Quantity"].astype(int)

    # 5. Handle Missing Values
    print("[STEP 5] Imputing and resolving missing values...")
    null_summary_before = df.isnull().sum()
    print("         Missing values before handling:\n", null_summary_before[null_summary_before > 0])

    # Date handling: Convert to datetime, drop records with invalid/unrecoverable dates
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="mixed", errors="coerce")
    missing_dates = df["Order_Date"].isnull().sum()
    if missing_dates > 0:
        print(f"         Dropping {missing_dates} records with missing/unparseable Order_Date.")
        df = df.dropna(subset=["Order_Date"]).copy()

    # Impute categorical nulls
    df["Customer_Segment"] = df["Customer_Segment"].fillna("Consumer")
    df["Payment_Method"] = df["Payment_Method"].fillna("Not Specified")

    # 6. Feature Engineering for Analytics & Power BI
    print("[STEP 6] Performing Feature Engineering & Financial Metric Re-calculation...")
    
    # Financial fields
    df["Gross_Sales"] = (df["Quantity"] * df["Unit_Price"]).round(2)
    df["Discount_Amount"] = (df["Gross_Sales"] * df["Discount"]).round(2)
    df["Net_Sales"] = (df["Gross_Sales"] - df["Discount_Amount"]).round(2)
    df["Total_Cost"] = (df["Quantity"] * df["Unit_Cost"]).round(2)
    df["Profit"] = (df["Net_Sales"] - df["Total_Cost"]).round(2)
    
    # Avoid division by zero
    df["Profit_Margin_Pct"] = np.where(
        df["Net_Sales"] > 0,
        ((df["Profit"] / df["Net_Sales"]) * 100).round(2),
        0.0
    )

    # Margin Category Segmentation
    conditions = [
        (df["Profit_Margin_Pct"] >= 25.0),
        (df["Profit_Margin_Pct"] >= 10.0) & (df["Profit_Margin_Pct"] < 25.0),
        (df["Profit_Margin_Pct"] >= 0.0) & (df["Profit_Margin_Pct"] < 10.0),
        (df["Profit_Margin_Pct"] < 0.0)
    ]
    choices = [
        "High Margin (>=25%)",
        "Moderate Margin (10-25%)",
        "Low Margin (0-10%)",
        "Loss Making (<0%)"
    ]
    df["Margin_Category"] = np.select(conditions, choices, default="Unclassified")

    # Date Dimension Attributes (Essential for Power BI Date Hierarchies & Time Intelligence)
    df["Year"] = df["Order_Date"].dt.year
    df["Month"] = df["Order_Date"].dt.month
    df["Month_Name"] = df["Order_Date"].dt.strftime("%b")
    df["Quarter"] = "Q" + df["Order_Date"].dt.quarter.astype(str)
    df["Day_of_Week"] = df["Order_Date"].dt.day_name()
    df["Year_Month"] = df["Order_Date"].dt.strftime("%Y-%m")

    # Sort chronological order for consistency
    df = df.sort_values(by="Order_Date").reset_index(drop=True)

    # 7. Final Quality Audit & Export
    final_rows, final_cols = df.shape
    print(f"[STEP 7] Cleaning complete.")
    print(f"         Initial records: {initial_rows:,} | Final Cleaned records: {final_rows:,}")
    print(f"         Removed records: {initial_rows - final_rows:,} ({(initial_rows - final_rows)/initial_rows:.1%})")
    print(f"         Columns expanded from {initial_cols} to {final_cols} analytics-ready features.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Cleaned Power BI-ready dataset exported to: {output_path}\n")

    return df


if __name__ == "__main__":
    clean_retail_data()
