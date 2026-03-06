"""Raw Data to Transactional Format Conversion"""

import pandas as pd
import pickle
from collections import Counter


def load_reduced_datasets():
    """Load all reduced datasets"""
    print("TRANSFORMING RAW DATA TO TRANSACTIONAL FORMAT")
    print("="*70)
    print("\nStep 1: Loading reduced datasets...")
    
    orders = pd.read_csv('orders_reduced.csv')
    order_products = pd.read_csv('order_products_reduced.csv')
    products = pd.read_csv('products_reduced.csv')
    aisles = pd.read_csv('aisles_reduced.csv')
    departments = pd.read_csv('departments_reduced.csv')
    
    print(f"  Loaded {len(orders):,} orders")
    print(f"  Loaded {len(order_products):,} order-product records")
    print(f"  Loaded {len(products):,} products")
    print(f"  Loaded {len(aisles):,} aisles")
    print(f"  Loaded {len(departments):,} departments")
    
    return orders, order_products, products, aisles, departments


def join_datasets(orders, order_products, products, aisles):
    """Join datasets to create enriched order items"""
    print("="*70)
    print("\nStep 2: Joining datasets to create enriched order items...")
    
    # Join order_products with products to get aisle_id
    order_items = order_products.merge(
        products[['product_id', 'aisle_id', 'product_name']],
        on='product_id',
        how='left'
    )
    
    # Join with aisles to get aisle names
    order_items = order_items.merge(
        aisles[['aisle_id', 'aisle']],
        on='aisle_id',
        how='left'
    )
    
    # Join with orders to get temporal information
    order_items = order_items.merge(
        orders[['order_id', 'order_dow', 'order_hour_of_day']],
        on='order_id',
        how='left'
    )
    
    print(f"  Created enriched dataset with {len(order_items):,} records")
    print(f"  Columns: {list(order_items.columns)}")
    
    return order_items


def get_time_period(hour):
    """Convert hour to time period"""
    if pd.isna(hour):
        return 'hour_unknown'
    elif 6 <= hour < 12:
        return 'time_morning'
    elif 12 <= hour < 14:
        return 'time_lunch'
    elif 14 <= hour < 17:
        return 'time_afternoon'
    elif 17 <= hour < 21:
        return 'time_evening'
    elif (21 <= hour < 24) or (0 <= hour < 1):
        return 'time_nightlife'
    else:
        return 'time_latenight'  # 1am-6am


def get_day_type(dow):
    """Convert day of week to day type"""
    if pd.isna(dow):
        return 'day_unknown'
    elif dow in [0, 6]:  # Sunday=0, Saturday=6
        return 'day_weekend'
    else:
        return 'day_weekday'


def create_item_representations(order_items):
    """Create item representations including temporal features"""
    print("="*70)
    print("\nStep 3: Creating item representations...")
    
    # Assuming aisle already lowercase, just replace spaces with underscores for better item names
    order_items['aisle_clean'] = (
        order_items['aisle'].str.replace(' ', '_')
    )
    
    # Create temporal features as items (categorical indicators)
    # Day of week: dow_0 to dow_6
    order_items['dow_item'] = 'dow_' + order_items['order_dow'].astype(str)
    
    # time-periods of the day
    order_items['time_period'] = order_items['order_hour_of_day'].apply(get_time_period)
    
    # Track day type (weekend vs weekday)
    #order_items['day_type'] = order_items['order_dow'].apply(get_day_type) # comment out as apriori maps dow_item to day type implicitly, so use 1 of 2
    
    print(f"  Created aisle items: {order_items['aisle_clean'].nunique()} unique aisles")
    print(f"  Created day-of-week items: {order_items['dow_item'].nunique()} unique values")
    print(f"  Created time period items: {order_items['time_period'].nunique()} unique values")
    #print(f"  Created day type items: {order_items['day_type'].nunique()} unique values")
    
    return order_items


def create_transactions(order_items):
    """Create transactions in basket format"""
    print("="*70)
    print("\nStep 4: Creating transactions (basket format)...")
    
    # For each order, collect all items (aisles + temporal features)
    transactions = []
    
    for order_id, group in order_items.groupby('order_id'):
        # Get unique aisle names for this order
        aisle_items = group['aisle_clean'].unique().tolist()
        
        # Get temporal features (same for all items in an order, so take first)
        dow_item = group['dow_item'].iloc[0]
        time_period = group['time_period'].iloc[0]
        #day_type = group['day_type'].iloc[0]
        
        # Combine all items into one transaction
        # Include: aisles + day of week + time period + day type
        transaction = aisle_items + [dow_item, time_period] #, day_type
        
        transactions.append(transaction)
    
    print(f"  Created {len(transactions):,} transactions")
    print(f"  Average items per transaction: {sum(len(t) for t in transactions) / len(transactions):.2f}")
    print(f"  Min items: {min(len(t) for t in transactions)}")
    print(f"  Max items: {max(len(t) for t in transactions)}")
    
    return transactions


def display_sample_transactions(transactions, num_samples=5):
    """Display sample transactions"""
    print("="*70)
    print(f"\nStep 5: Display sample transactions...")
    print(f"\nFirst {num_samples} transactions:")
    for i, transaction in enumerate(transactions[:num_samples], 1):
        print(f"\n  Transaction {i} ({len(transaction)} items):")
        # Separate aisles from temporal features for better readability
        aisles_in_txn = [item for item in transaction if not item.startswith(('dow_', 'time_', 'day_'))]
        temporal_in_txn = [item for item in transaction if item.startswith(('dow_', 'time_', 'day_'))]
        print(f"    Aisles: {aisles_in_txn[:10]}{'...' if len(aisles_in_txn) > 10 else ''}")
        print(f"    Temporal: {temporal_in_txn}")


def save_transactions(transactions):
    """Save transactions in multiple formats"""
    print("\n" + "="*70)
    print("Step 6: Saving transactions...")
    
    # Format 1: Pickle file (preserves list of lists for Python)
    with open('transactions.pkl', 'wb') as f:
        pickle.dump(transactions, f)
    print("  Saved: transactions.pkl (Python pickle format)")
    
    # Format 2: CSV file (one row per transaction, comma-separated items)
    with open('transactions.csv', 'w', encoding='utf-8') as f:
        for transaction in transactions:
            f.write(','.join(transaction) + '\n')
    print("  Saved: transactions.csv (CSV format)")


def print_transaction_summary(transactions):
    """Print transaction dataset summary and statistics"""
    print("\n" + "="*70)
    print("TRANSACTION DATASET SUMMARY")
    print("="*70)
    
    # Count all unique items
    all_items = set()
    for transaction in transactions:
        all_items.update(transaction)
    
    # Separate item types
    aisle_items = {item for item in all_items if not item.startswith(('dow_', 'time_', 'day_'))}
    temporal_items = {item for item in all_items if item.startswith(('dow_', 'time_', 'day_'))}
    
    print(f"\nTotal transactions: {len(transactions):,}")
    print(f"Total unique items: {len(all_items)}")
    print(f"  - Aisle items: {len(aisle_items)}")
    print(f"  - Temporal items: {len(temporal_items)}")
    
    print(f"\nTransaction statistics:")
    transaction_lengths = [len(t) for t in transactions]
    print(f"  - Average items per transaction: {sum(transaction_lengths) / len(transactions):.2f}")
    print(f"  - Min items: {min(transaction_lengths)}")
    print(f"  - Max items: {max(transaction_lengths)}")
    print(f"  - Median items: {sorted(transaction_lengths)[len(transaction_lengths)//2]}")
    
    # Item frequency analysis
    all_items_flat = [item for transaction in transactions for item in transaction]
    item_counts = Counter(all_items_flat)
    
    print(f"\nTop 10 most frequent items:")
    for item, count in item_counts.most_common(10):
        support = count / len(transactions) * 100
        print(f"  {item}: {count:,} times ({support:.2f}% support)")
    
    print("\n" + "="*70)
    print("Transactions generation done")


def run():
    """
    Main function to run the transaction generation pipeline
    """
    # Load reduced datasets
    orders, order_products, products, aisles, departments = load_reduced_datasets()
    
    # Join datasets to create enriched order items
    order_items = join_datasets(orders, order_products, products, aisles)
    
    # Create item representations (aisles + temporal features)
    order_items = create_item_representations(order_items)
    
    # Create transactions in basket format
    transactions = create_transactions(order_items)
    
    # Display sample transactions
    display_sample_transactions(transactions)
    
    # Save transactions
    save_transactions(transactions)
    
    # Print summary statistics
    print_transaction_summary(transactions)
    
    return transactions


if __name__ == "__main__":
    run()

