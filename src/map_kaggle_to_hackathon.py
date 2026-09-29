import pandas as pd
import numpy as np
import os

base_dir = r"d:\college\DS hackathon\STOCKSENSE_Team Apex"
raw_dir = os.path.join(base_dir, "data", "raw")
out_dir = os.path.join(base_dir, "data", "processed")

print("1. Loading Kaggle sample data...")
try:
    # 1. Stores Mapping
    df_stores_k = pd.read_csv(os.path.join(raw_dir, "stores.csv"))
    df_stores = pd.DataFrame()
    df_stores['store_id'] = "S" + df_stores_k['store_nbr'].astype(str)
    df_stores['city'] = df_stores_k['city']
    df_stores['store_type'] = df_stores_k['type']
    df_stores['floor_area_sqft'] = np.random.randint(4000, 20000, size=len(df_stores))
    df_stores['avg_daily_customers'] = np.random.randint(500, 3000, size=len(df_stores))
    df_stores['region'] = df_stores_k['state']
    df_stores.to_csv(os.path.join(out_dir, "stores.csv"), index=False)
    print(" - Created stores.csv")

    # 2. Products Mapping
    df_items_k = pd.read_csv(os.path.join(raw_dir, "items.csv"))
    df_prods = pd.DataFrame()
    df_prods['product_id'] = "P" + df_items_k['item_nbr'].astype(str)
    df_prods['category'] = df_items_k['family']
    df_prods['sub_category'] = df_items_k['class'].astype(str)
    df_prods['brand'] = "Brand_" + np.random.randint(1, 50, size=len(df_prods)).astype(str)
    df_prods['mrp'] = np.round(np.random.uniform(20, 500, size=len(df_prods)), 2)
    df_prods['cost_price'] = np.round(df_prods['mrp'] * np.random.uniform(0.5, 0.8, size=len(df_prods)), 2)
    df_prods['shelf_life_days'] = np.random.choice([3, 7, 30, 90, 180, 365, 730], size=len(df_prods))
    df_prods['supplier_id'] = "SUP" + pd.Series(np.random.randint(1, 20, size=len(df_prods))).astype(str).str.zfill(2)
    df_prods.to_csv(os.path.join(out_dir, "products.csv"), index=False)
    print(" - Created products.csv")

    # 3. Transactions Mapping (Taking a 50,000 row sample of the 5GB train.csv)
    print(" - Sampling train.csv (50,000 rows)...")
    df_train_k = pd.read_csv(os.path.join(raw_dir, "train.csv"), nrows=50000)
    df_trans = pd.DataFrame()
    df_trans['transaction_id'] = ["T" + str(100000 + i) for i in range(len(df_train_k))]
    df_trans['date'] = df_train_k['date']
    df_trans['store_id'] = "S" + df_train_k['store_nbr'].astype(str)
    df_trans['product_id'] = "P" + df_train_k['item_nbr'].astype(str)
    df_trans['quantity'] = np.where(df_train_k['unit_sales'] < 0, 0, df_train_k['unit_sales']).astype(int)
    
    # Merge mrp to calculate selling price based on discount
    df_trans = df_trans.merge(df_prods[['product_id', 'mrp']], on='product_id', how='left')
    # If a product_id wasn't in the item lookup, give it a default MRP
    df_trans['mrp'] = df_trans['mrp'].fillna(100.0) 
    
    df_trans['promotion_flag'] = np.where(df_train_k['onpromotion'] == True, 1, 0)
    df_trans['discount_pct'] = np.where(df_trans['promotion_flag'] == 1, np.random.choice([5.0, 10.0, 15.0, 20.0], size=len(df_trans)), 0.0)
    df_trans['selling_price'] = np.round(df_trans['mrp'] * (1 - df_trans['discount_pct']/100), 2)
    df_trans['customer_id'] = "C" + np.random.randint(1000, 9000, size=len(df_trans)).astype(str)
    df_trans['payment_mode'] = np.random.choice(['UPI', 'Card', 'Cash'], size=len(df_trans), p=[0.5, 0.3, 0.2])
    df_trans['hour'] = np.random.randint(8, 22, size=len(df_trans))
    df_trans.drop(columns=['mrp'], inplace=True)
    df_trans.to_csv(os.path.join(out_dir, "transactions.csv"), index=False)
    print(" - Created transactions.csv")

    # 4. Inventory Mapping (Synthesized from transactions)
    df_sales = df_trans.groupby(['date', 'store_id', 'product_id'])['quantity'].sum().reset_index()
    df_sales.rename(columns={'quantity': 'sold'}, inplace=True)
    df_sales['opening'] = df_sales['sold'] + np.random.randint(10, 100, size=len(df_sales))
    df_sales['received'] = np.random.choice([0, 50, 100], size=len(df_sales), p=[0.7, 0.2, 0.1])
    df_sales['closing'] = df_sales['opening'] + df_sales['received'] - df_sales['sold']
    # Intentional Data Trap: Let's create a few inventory arithmetic mismatches!
    trap_idx = np.random.choice(df_sales.index, size=int(len(df_sales)*0.02), replace=False)
    df_sales.loc[trap_idx, 'closing'] = df_sales.loc[trap_idx, 'closing'] + np.random.randint(5, 20, size=len(trap_idx))
    
    df_sales['reorder_lvl'] = np.random.randint(20, 80, size=len(df_sales))
    df_sales['lead_days'] = np.random.randint(1, 7, size=len(df_sales))
    df_sales.rename(columns={'store_id': 'store', 'product_id': 'product'}, inplace=True)
    df_sales.to_csv(os.path.join(out_dir, "inventory.csv"), index=False)
    print(" - Created inventory.csv (with intentional data traps!)")

    # 5. External Factors Mapping
    df_hol_k = pd.read_csv(os.path.join(raw_dir, "holidays_events.csv"))
    unique_dates = df_trans['date'].unique()
    unique_cities = df_stores['city'].unique()
    idx = pd.MultiIndex.from_product([unique_dates, unique_cities], names=['date', 'city'])
    df_ext = pd.DataFrame(index=idx).reset_index()
    
    df_ext['temp_c'] = np.round(np.random.uniform(22.0, 38.0, size=len(df_ext)), 1)
    # Intentional Data Trap: Missing values in temperature
    null_idx = np.random.choice(df_ext.index, size=int(len(df_ext)*0.05), replace=False)
    df_ext.loc[null_idx, 'temp_c'] = np.nan
    
    df_ext['rain_mm'] = np.round(np.random.exponential(2, size=len(df_ext)), 1)
    holidays_dates = df_hol_k['date'].unique()
    df_ext['holiday'] = np.where(df_ext['date'].isin(holidays_dates), 1, 0)
    df_ext['festival'] = np.where(df_ext['holiday'] == 1, np.random.choice([0, 1], size=len(df_ext), p=[0.7, 0.3]), 0)
    df_ext['date_dt'] = pd.to_datetime(df_ext['date'])
    df_ext['weekend'] = np.where(df_ext['date_dt'].dt.dayofweek >= 5, 1, 0)
    df_ext['local_event'] = np.random.choice([0, 1], size=len(df_ext), p=[0.9, 0.1])
    df_ext.drop(columns=['date_dt'], inplace=True)
    
    df_ext.to_csv(os.path.join(out_dir, "external_factors.csv"), index=False)
    print(" - Created external_factors.csv")

    print("\nSUCCESS: All 5 hackathon files generated successfully in data/processed!")

except Exception as e:
    print(f"ERROR: {e}")
