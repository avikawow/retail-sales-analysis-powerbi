"""
==============================================================================
Project: Retail Sales Analysis
Module: generate_raw_data.py
Description: Generates a realistic synthetic raw retail sales dataset with 
             intentional real-world data quality issues (duplicates, nulls, 
             format inconsistencies, whitespace) for demonstration of data 
             cleaning, preprocessing, and exploratory data analysis.
Note: All data is synthetic and does not represent any real-world commercial entity.
==============================================================================
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_retail_data(
    num_records: int = 2800,
    random_seed: int = 42,
    output_path: str = "data/raw/retail_sales_raw.csv"
) -> pd.DataFrame:
    """
    Generates a realistic raw retail transactions dataset and saves it to CSV.

    Parameters:
    -----------
    num_records : int
        Number of base records to generate.
    random_seed : int
        Seed for reproducibility.
    output_path : str
        Filepath where the raw CSV will be written.

    Returns:
    --------
    pd.DataFrame: The simulated raw DataFrame.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Product catalog: Category -> Subcategory -> (Product_Name, Base_Price, Base_Cost)
    catalog = [
        # Electronics
        ("Electronics", "Smartphones", "Smartphone Pro 128GB", 799.00, 610.00),
        ("Electronics", "Laptops & PCs", "Ultra-Slim 14-inch Laptop", 950.00, 740.00),
        ("Electronics", "Audio", "Noise-Canceling Wireless Headphones", 179.00, 110.00),
        ("Electronics", "Audio", "Bluetooth Portable Speaker", 49.00, 26.00),
        ("Electronics", "Television", "Ultra HD Smart LED TV 55-inch", 649.00, 490.00),
        ("Electronics", "Accessories", "USB-C Multiport Fast Hub", 34.00, 15.00),
        ("Electronics", "Wearables", "Smart Fitness Health Watch", 129.00, 75.00),

        # Clothing & Apparel
        ("Clothing", "Men's Apparel", "Men's Slim Fit Denim Jeans", 49.00, 22.00),
        ("Clothing", "Women's Apparel", "Women's Floral Cotton Dress", 59.00, 26.00),
        ("Clothing", "Footwear", "Breathable Mesh Running Shoes", 79.00, 39.00),
        ("Clothing", "Men's Apparel", "Classic Organic Cotton T-Shirt", 22.00, 8.50),
        ("Clothing", "Outerwear", "Fleece Zip Hooded Sweatshirt", 45.00, 21.00),
        ("Clothing", "Clearance", "Budget Graphic Polyester Tee", 14.00, 13.50),  # Low/negative margin underperformer

        # Home & Kitchen
        ("Home & Kitchen", "Cookware", "Stainless Steel 10-Piece Cookware Set", 169.00, 112.00),
        ("Home & Kitchen", "Small Appliances", "Digital Air Fryer 5.8 Qt", 89.00, 56.00),
        ("Home & Kitchen", "Coffee Makers", "Automatic Espresso & Cappuccino Maker", 229.00, 155.00),
        ("Home & Kitchen", "Bedding", "Orthopedic Memory Foam Pillow", 39.00, 18.00),
        ("Home & Kitchen", "Cleaning", "Robot Smart Vacuum Cleaner", 279.00, 215.00),
        ("Home & Kitchen", "Storage", "Glass Food Storage Containers (Set of 6)", 28.00, 13.00),

        # Beauty & Personal Care
        ("Beauty & Personal Care", "Skincare", "Hydrating Hyaluronic Acid Serum", 29.00, 9.50),
        ("Beauty & Personal Care", "Haircare", "Argan Oil Repair Shampoo & Conditioner", 24.00, 8.50),
        ("Beauty & Personal Care", "Oral Care", "Rechargeable Sonic Electric Toothbrush", 69.00, 34.00),
        ("Beauty & Personal Care", "Skincare", "Revitalizing Night Face Cream", 35.00, 12.00),
        ("Beauty & Personal Care", "Hair Styling", "Ionic Salon Hair Dryer", 75.00, 36.00),

        # Sports & Outdoors
        ("Sports & Outdoors", "Fitness", "Eco-Friendly Yoga Mat with Strap", 32.00, 14.00),
        ("Sports & Outdoors", "Strength Training", "Adjustable Quick-Select Dumbbell Set", 159.00, 102.00),
        ("Sports & Outdoors", "Camping", "Waterproof 4-Person Camping Tent", 139.00, 88.00),
        ("Sports & Outdoors", "Hydration", "Insulated Vacuum Water Bottle 1L", 24.00, 9.50),
        ("Sports & Outdoors", "Outdoor Gear", "Tactical Multi-Pocket Hiking Backpack", 68.00, 36.00)
    ]

    regions = ["North", "South", "East", "West", "Central"]
    customer_segments = ["Consumer", "Corporate", "Home Office"]
    payment_methods = ["Credit Card", "Debit Card", "UPI / Digital Wallet", "Cash on Delivery", "Net Banking"]
    shipping_statuses = ["Delivered", "Delivered", "Delivered", "Shipped", "Pending", "Cancelled"]

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_delta = (end_date - start_date).days

    records = []

    for i in range(1, num_records + 1):
        txn_id = f"TXN-{10000 + i}"
        cust_id = f"CUST-{random.randint(1001, 1999)}"

        # Realistic seasonal weighting: Q4 has higher sales volume
        random_day = random.randint(0, date_delta)
        order_date = start_date + timedelta(days=random_day)

        # Monthly seasonal bump in Q4 (Oct, Nov, Dec)
        month = order_date.month
        if month in [10, 11, 12] and random.random() < 0.35:
            # Pick again to bias towards Q4
            order_date = start_date + timedelta(days=random.randint(273, date_delta))

        # Select product with weighted popularity
        prod_tuple = random.choice(catalog)
        category, subcategory, product_name, unit_price, unit_cost = prod_tuple

        # Quantity: most orders are 1-3 units, rarely 4-8
        qty_weights = [0.45, 0.30, 0.15, 0.05, 0.03, 0.01, 0.01]
        quantity = random.choices([1, 2, 3, 4, 5, 6, 8], weights=qty_weights)[0]

        # Discounts: clearance gets high discount, electronics low, holidays higher
        if "Clearance" in subcategory:
            discount = random.choice([0.25, 0.30, 0.40, 0.50])
        elif order_date.month in [11, 12]:  # Holiday sales
            discount = random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
        else:
            discount = random.choice([0.0, 0.0, 0.05, 0.10, 0.15])

        region = random.choice(regions)
        segment = random.choice(customer_segments)
        payment = random.choice(payment_methods)
        shipping = random.choice(shipping_statuses)

        # Base calculations
        gross_sales = quantity * unit_price
        discount_amount = gross_sales * discount
        net_sales = round(gross_sales - discount_amount, 2)
        total_cost = round(quantity * unit_cost, 2)
        profit = round(net_sales - total_cost, 2)

        # Introduce realistic raw data quirks:
        # 1. Date formatting variations (some string formats, some with dashes or slashes)
        date_roll = random.random()
        if date_roll < 0.015:
            date_str = None  # Missing date
        elif date_roll < 0.15:
            date_str = order_date.strftime("%m/%d/%Y")  # US format MM/DD/YYYY
        elif date_roll < 0.25:
            date_str = order_date.strftime("%d-%m-%Y")  # DD-MM-YYYY
        else:
            date_str = order_date.strftime("%Y-%m-%d")  # ISO format

        # 2. Text quirks (extra spaces, inconsistent casing)
        region_str = region
        if random.random() < 0.10:
            region_str = f"  {region.lower()}  " if random.random() < 0.5 else region.upper()

        category_str = category
        if random.random() < 0.08:
            category_str = f" {category} "

        # 3. Unit price formatting quirk (some rows formatted as strings with "$")
        price_val = unit_price
        if random.random() < 0.05:
            price_val = f"${unit_price:.2f}"

        # 4. Missing categorical values
        segment_val = segment
        if random.random() < 0.02:
            segment_val = None

        payment_val = payment
        if random.random() < 0.02:
            payment_val = None

        # 5. Occasional invalid quantity or missing values
        qty_val = quantity
        if random.random() < 0.008:
            qty_val = -1 if random.random() < 0.5 else 0

        records.append({
            "Transaction_ID": txn_id,
            "Order_Date": date_str,
            "Customer_ID": cust_id,
            "Customer_Segment": segment_val,
            "Region": region_str,
            "Product_Category": category_str,
            "Product_Subcategory": subcategory,
            "Product_Name": product_name,
            "Unit_Price": price_val,
            "Unit_Cost": unit_cost,
            "Quantity": qty_val,
            "Discount": discount,
            "Sales": net_sales,
            "Profit": profit,
            "Payment_Method": payment_val,
            "Shipping_Status": shipping
        })

    df = pd.DataFrame(records)

    # 6. Add exact duplicate rows (~30 duplicates) to test duplicate detection
    duplicate_rows = df.sample(n=32, random_state=random_seed)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # Shuffle the dataset
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Ensure target output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Synthetic raw retail dataset generated: {output_path}")
    print(f"Total Rows: {len(df)}, Total Columns: {len(df.columns)}")
    return df

if __name__ == "__main__":
    generate_synthetic_retail_data()
