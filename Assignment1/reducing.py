"""Data Reduction Script"""

import pandas as pd
import numpy as np


def load_datasets():
    """Load all original datasets"""
    print("Loading datasets...")
    
    data_folder = "data/"
    orders = pd.read_csv(data_folder+'orders.csv')
    order_products = pd.read_csv(data_folder+'order_products.csv')
    products = pd.read_csv(data_folder+'products.csv')
    aisles = pd.read_csv(data_folder+'aisles.csv')
    departments = pd.read_csv(data_folder+'departments.csv')
    
    print(f"Original dataset sizes:")
    print(f"  Orders: {len(orders):,} orders")
    print(f"  Order-Products: {len(order_products):,} records")
    print(f"  Products: {len(products):,} products")
    print(f"  Aisles: {len(aisles):,} aisles")
    print(f"  Departments: {len(departments):,} departments")
    
    return orders, order_products, products, aisles, departments


def filter_rare_products(order_products, min_product_purchases=30):
    """Filter out products purchased less than min_product_purchases times"""
    print("="*70)
    print(f"STEP 1: Filtering rare products (purchased < {min_product_purchases} times)")
    
    # Count how many times each product was purchased
    product_counts = order_products['product_id'].value_counts()
    print(f"Total unique products in orders: {len(product_counts):,}")
    
    frequent_products = product_counts[product_counts >= min_product_purchases].index
    print(f"Products purchased >= {min_product_purchases} times: {len(frequent_products):,}")
    print(f"Products removed: {len(product_counts) - len(frequent_products):,}")
    
    # Filter order_products to keep only frequent products
    order_products_filtered = order_products[
        order_products['product_id'].isin(frequent_products)
    ].copy()
    
    print(f"Order-Product records after filtering: {len(order_products_filtered):,}")
    print(f"Records removed: {len(order_products) - len(order_products_filtered):,}")
    
    return order_products_filtered


def sample_orders(orders, order_products_filtered, sample_rate=0.1, random_seed=42):
    """Sample a percentage of orders and filter related data"""
    print("="*70)
    print(f"STEP 2: Sampling {sample_rate*100:.0f}% of orders")
    
    np.random.seed(random_seed)
    
    unique_orders = orders['order_id'].unique()
    sample_size = int(len(unique_orders) * sample_rate)
    sampled_order_ids = np.random.choice(unique_orders, size=sample_size, replace=False)
    
    print(f"Total orders: {len(unique_orders):,}")
    print(f"Sampled orders ({sample_rate*100:.0f}%): {len(sampled_order_ids):,}")
    
    # Filter orders to keep only sampled orders
    orders_sampled = orders[orders['order_id'].isin(sampled_order_ids)].copy()
    
    # Filter order_products to keep only sampled orders
    order_products_final = order_products_filtered[
        order_products_filtered['order_id'].isin(sampled_order_ids)
    ].copy()
    
    print(f"Final order-product records: {len(order_products_final):,}")
    
    return orders_sampled, order_products_final


def filter_products_and_metadata(order_products_final, products, aisles, departments):
    """Filter products, aisles, and departments to match final order data"""
    # Get list of products that appear in the final dataset
    final_product_ids = order_products_final['product_id'].unique()
    products_final = products[products['product_id'].isin(final_product_ids)].copy()
    
    print(f"Products in final dataset: {len(products_final):,}")
    
    # Keep only used aisles and departments
    used_aisle_ids = products_final['aisle_id'].unique()
    used_dept_ids = products_final['department_id'].unique()
    
    aisles_final = aisles[aisles['aisle_id'].isin(used_aisle_ids)].copy()
    departments_final = departments[departments['department_id'].isin(used_dept_ids)].copy()
    
    return products_final, aisles_final, departments_final


def save_reduced_datasets(orders_sampled, order_products_final, products_final, 
                         aisles_final, departments_final):
    """Save all reduced datasets to CSV files"""
    print("\n" + "="*70)
    print("STEP 3: Saving reduced datasets")
    
    reduced_data_folder = './data_reduced/'
    orders_sampled.to_csv(reduced_data_folder+'orders_reduced.csv', index=False)
    order_products_final.to_csv(reduced_data_folder+'order_products_reduced.csv', index=False)
    products_final.to_csv(reduced_data_folder+'products_reduced.csv', index=False)
    aisles_final.to_csv(reduced_data_folder+'aisles_reduced.csv', index=False)
    departments_final.to_csv(reduced_data_folder+'departments_reduced.csv', index=False)
    
    print("Saved: data_reduced/orders_reduced.csv")
    print("Saved: data_reduced/order_products_reduced.csv") 
    print("Saved: data_reduced/products_reduced.csv")
    print("Saved: data_reduced/aisles_reduced.csv")
    print("Saved: data_reduced/departments_reduced.csv")


def print_summary(orders, order_products, products, aisles, departments,
                 orders_sampled, order_products_final, products_final, 
                 aisles_final, departments_final):
    """Print reduction summary statistics"""
    print("="*70)
    print("REDUCTION SUMMARY")
    
    reduction_stats = {
        'Dataset': ['Orders', 'Order-Products', 'Products', 'Aisles', 'Departments'],
        'Original': [
            len(orders),
            len(order_products),
            len(products),
            len(aisles),
            len(departments)
        ],
        'Reduced': [
            len(orders_sampled),
            len(order_products_final),
            len(products_final),
            len(aisles_final),
            len(departments_final)
        ]
    }
    
    stats_df = pd.DataFrame(reduction_stats)
    stats_df['Reduction %'] = (
        (stats_df['Original'] - stats_df['Reduced']) / stats_df['Original'] * 100
    ).round(2)
    
    print(stats_df.to_string(index=False))
    print("\nDataset reduction done")


def run(min_product_purchases=30, sample_rate=0.1, random_seed=42):
    """
    Run the data reduction pipeline
    
    Args:
        min_product_purchases: Minimum number of purchases for a product to be included
        sample_rate: Fraction of orders to sample (0.0 to 1.0)
        random_seed: Random seed for reproducibility
    """
    orders, order_products, products, aisles, departments = load_datasets()
    
    order_products_filtered = filter_rare_products(order_products, min_product_purchases)
    
    orders_sampled, order_products_final = sample_orders(
        orders, order_products_filtered, sample_rate, random_seed
    )
    
    products_final, aisles_final, departments_final = filter_products_and_metadata(
        order_products_final, products, aisles, departments
    )
    
    save_reduced_datasets(
        orders_sampled, order_products_final, products_final, 
        aisles_final, departments_final
    )
    
    print_summary(
        orders, order_products, products, aisles, departments,
        orders_sampled, order_products_final, products_final, 
        aisles_final, departments_final
    )


if __name__ == "__main__":
    run()
