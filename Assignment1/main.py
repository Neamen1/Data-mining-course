"""
This script runs the full data mining pipeline:
1. Data Reduction (reducing.py)
2. Transaction Format Conversion (raw_to_transactions.py)
3. Association Rule Mining (associations.py)
"""

import sys
import time
from datetime import datetime
import os
from pathlib import Path

# Import our modules
import reducing
import raw_to_transactions
import associations

def print_header(text):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {text}")

def print_step(step_num, total_steps, description):
    """Print step information"""
    print(f"\n{'='*80}")
    print(f"STEP {step_num}/{total_steps}: {description}")

def main(reducing_params, transaction_params, association_params):
    """Main pipeline execution"""
    
    # Record start time
    pipeline_start = time.time()
    
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis pipeline will execute:")
    print("  1. Data Reduction (reducing.py)")
    print("  2. Transaction Format Conversion (raw_to_transactions.py)")
    print("  3. Association Rule Mining (associations.py)")
    
    input("\nPress Enter to start the pipeline...")
    
    print_step(1, 3, "DATA REDUCTION")
    print("Reducing dataset size using hybrid strategy:")
    print("  - Remove products purchased < 30 times")
    print(f"  - Sample {reducing_params.get('sample_rate', 0.5)*100:.0f}% of orders")
    print()
    
    start_time = time.time()
    
    try:
        # Call the function with respective parameters
        reducing.run(**reducing_params)
        elapsed_time = time.time() - start_time
        print(f"\nData reduction completed successfully in {elapsed_time:.2f} seconds")
    except Exception as e:
        elapsed_time = time.time() - start_time
        import traceback
        traceback.print_exc()
        raise Exception(f"Error running data reduction after {elapsed_time} seconds: {str(e)}")
    
    print_step(2, 3, "TRANSACTION FORMAT CONVERSION")
    print("Converting raw data to transactional format:")
    print("  - Transaction = One order")
    print("  - Items = Aisle names + temporal features")
    print("  - Format = Basket format for Apriori")
    print()
    start_time = time.time()
    try:
        # Call the function with respective parameters (for transactions should be empty in current implementation)
        raw_to_transactions.run(**transaction_params)
        elapsed_time = time.time() - start_time
        print(f"\nTransaction format conversion completed successfully in {elapsed_time:.2f} seconds")
    except Exception as e:
        elapsed_time = time.time() - start_time
        import traceback
        traceback.print_exc()
        raise Exception(f"Error running transaction conversion after {elapsed_time} seconds: {str(e)}")
    

    print_step(3, 3, "ASSOCIATION RULE MINING")
    print("Mining association rules using Apriori:")
    print("  - Parameter experimentation")
    print("  - Product category analysis")
    print("  - Purchase timing analysis")
    print()
    
    start_time = time.time()
    try:
        # Call the function with respective parameters
        associations.run(**association_params)
        elapsed_time = time.time() - start_time
        print(f"\nAssociation rule mining completed successfully in {elapsed_time:.2f} seconds")
    except Exception as e:
        elapsed_time = time.time() - start_time
        import traceback
        traceback.print_exc()
        raise Exception(f"Error running association mining after {elapsed_time} seconds: {str(e)}")
    

    pipeline_end = time.time()
    total_time = pipeline_end - pipeline_start
    
    print_header("PIPELINE SUMMARY")
        
    print("\n" + "-" * 80)
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    print("\n" + "="*80)
    print("  PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\nGenerated Files:")
    print("-" * 80)
    print("Data Files:")
    print("  - orders_reduced.csv - Reduced orders")
    print("  - order_products_reduced.csv - Reduced order-products")
    print("  - products_reduced.csv - Reduced products")
    print("  - aisles_reduced.csv - Reduced aisles")
    print("  - departments_reduced.csv - Reduced departments")
    print("  - transactions.pkl - Transaction database (pickle)")
    print("  - transactions.csv - Transaction database (CSV)")
    
    print("\nAnalysis Results:")
    print("  - parameter_experiments.csv - Parameter testing results")
    print("  - association_rules_all.csv - All association rules")
    print("  - rules_product_categories.csv - Product category rules")
    print("  - rules_purchase_timing.csv - Temporal pattern rules")
    
    print("\nVisualizations:")
    print("  - parameter_effects.png - Parameter effect analysis")
    print("  - parameter_effects_lift.png - Lift analysis")
    print("  - top_rules_visualization.png - Top rules visualization")        

if __name__ == "__main__":
    # change working directory to the script's directory to ensure relative paths work correctly
    os.chdir(Path(__file__).parent.resolve())

    try:
        reducing_params = {
            'min_product_purchases': 30,
            'sample_rate': 0.5,     # Sample x% of orders
            'random_seed': 42
        }
        transaction_params = {} # No parameters needed for current transaction implementation
        association_params = {
            'min_support': 0.05,
            'min_confidence': 0.3,
            'min_lift': 1.2,
            'do_parameter_experimentation': True  # Set to True to run parameter experimentation
        }
        main(reducing_params, transaction_params, association_params)
    except KeyboardInterrupt:
        print("\n\n Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n Unexpected error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)