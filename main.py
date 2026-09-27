"""
==============================================================================
Project: Retail Sales Analysis
Module: main.py
Description: Master end-to-end pipeline orchestrator for the Retail Sales 
             Analysis portfolio project. Executes data generation, cleaning,
             exploratory data analysis, and chart generation in sequence.
==============================================================================
"""

import sys
import os
import argparse
from src.generate_raw_data import generate_synthetic_retail_data
from src.data_cleaning import clean_retail_data
from src.eda_analysis import run_full_eda
from src.generate_visualizations import generate_all_visualizations


def run_pipeline(force_generate: bool = False):
    """
    Executes the complete retail analytics workflow.
    """
    print("\n" + "=" * 75)
    print("      RETAIL SALES ANALYSIS - END-TO-END ANALYTICS PIPELINE")
    print("=" * 75)

    raw_path = "data/raw/retail_sales_raw.csv"
    clean_path = "data/processed/retail_sales_cleaned.csv"
    figures_dir = "reports/figures"

    # Step 1: Raw Data Check / Generation
    if force_generate or not os.path.exists(raw_path):
        print("\n>>> PIPELINE STEP 1: Generating realistic synthetic retail dataset...")
        generate_synthetic_retail_data(num_records=2800, output_path=raw_path)
    else:
        print(f"\n>>> PIPELINE STEP 1: Raw data already exists at '{raw_path}'. (Use --generate to rebuild)")

    # Step 2: Data Cleaning & Preprocessing
    print("\n>>> PIPELINE STEP 2: Running Data Cleaning & Preprocessing...")
    clean_retail_data(input_path=raw_path, output_path=clean_path)

    # Step 3: Exploratory Data Analysis & KPIs
    print("\n>>> PIPELINE STEP 3: Executing Exploratory Data Analysis (EDA)...")
    run_full_eda(filepath=clean_path)

    # Step 4: High-Resolution Data Visualizations
    print("\n>>> PIPELINE STEP 4: Creating publication-ready visualizations...")
    generate_all_visualizations(csv_path=clean_path, output_dir=figures_dir)

    print("=" * 75)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"1. Raw Dataset                : {raw_path}")
    print(f"2. Cleaned Power BI Dataset   : {clean_path}")
    print(f"3. Generated Visualizations   : {figures_dir}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail Sales Analysis Pipeline")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Force regeneration of the synthetic raw dataset"
    )
    args = parser.parse_args()
    run_pipeline(force_generate=args.generate)
