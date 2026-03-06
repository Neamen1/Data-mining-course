"""
Association Rule Mining using Apriori Algorithm

It includes:
1. Parameter testing (min_support, min_confidence) & results visualization
2. Analysis for different objectives:
   - Product Categories (aisle associations)
   - Purchase Timing (temporal patterns)
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from apyori import apriori
import numpy as np

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_transactions(file_path='transactions.pkl'):
    """Load transaction data from pickle file"""
    print("="*80)
    print("ASSOCIATION RULE MINING - APRIORI ALGORITHM")
    print("="*80)
    
    print("\nStep 1: Loading transactions...")
    
    with open(file_path, 'rb') as f:
        transactions = pickle.load(f)
    
    print(f"  Loaded {len(transactions):,} transactions")
    
    # Get all unique items
    all_items = set()
    for transaction in transactions:
        all_items.update(transaction)
    print(f"  Total unique items: {len(all_items)}")
    
    return transactions


def parameter_experimentation(transactions, support_values=[0.04, 0.08, 0.15], confidence_values=[0.2, 0.4, 0.6]):
    """Test various parameter combinations and visualize results."""
    print("\n" + "="*80)
    print("STEP 2.1: PARAMETER EXPERIMENTATION")
    print("="*80)
    print("\nTesting various min_support and min_confidence values...")
        
    # Store results
    experiment_results = []
    
    for min_sup in support_values:
        for min_conf in confidence_values:
            print(f"Testing: support={min_sup:.3f}, confidence={min_conf:.2f}...")
            
            # Run Apriori
            rules = list(apriori(
                transactions,
                min_support=min_sup,
                min_confidence=min_conf,
                min_lift=1.0,
                min_length=2,
                max_length=10
            ))
            
            # Count rules
            num_rules = len(rules)
            
            # Calculate average confidence and lift
            if num_rules > 0:
                confidences = []
                lifts = []
                for rule in rules:
                    for stat in rule.ordered_statistics:
                        confidences.append(stat.confidence)
                        lifts.append(stat.lift)
                
                avg_confidence = np.mean(confidences) if confidences else 0
                avg_lift = np.mean(lifts) if lifts else 0
            else:
                avg_confidence = 0
                avg_lift = 0
            
            experiment_results.append({
                'min_support': min_sup,
                'min_confidence': min_conf,
                'num_rules': num_rules,
                'avg_confidence': avg_confidence,
                'avg_lift': avg_lift
            })
            
            print(f"{num_rules} rules")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(experiment_results)
    results_df.to_csv('parameter_experiments.csv', index=False)
    print(f"\n  Saved experiment results to parameter_experiments.csv")
    
    # Visualize results
    print("\n" + "="*80)
    print("STEP 2.2: VISUALIZING PARAMETER EFFECTS")
    print("="*80)
    
    # Create pivot tables for heatmaps
    pivot_num_rules = results_df.pivot(
        index='min_support',
        columns='min_confidence',
        values='num_rules'
    )
    
    # Plot 1: Number of rules for different parameter combinations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap of number of rules
    sns.heatmap(pivot_num_rules, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0])
    axes[0].set_title('Number of Association Rules Generated', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Minimum Confidence', fontsize=12)
    axes[0].set_ylabel('Minimum Support', fontsize=12)
    
    # Line plot showing effect of support threshold
    for conf in [0.2, 0.4, 0.6]:
        subset = results_df[results_df['min_confidence'] == conf]
        axes[1].plot(subset['min_support'], subset['num_rules'], 
                    marker='o', label=f'confidence={conf:.1f}', linewidth=2)
    
    axes[1].set_xlabel('Minimum Support', fontsize=12)
    axes[1].set_ylabel('Number of Rules', fontsize=12)
    axes[1].set_title('Effect of Support Threshold on Rule Count', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_effects.png', dpi=300, bbox_inches='tight')
    print("  Saved visualization: parameter_effects.png")
    
    # Plot 2: Effect on average lift
    pivot_avg_lift = results_df.pivot(
        index='min_support',
        columns='min_confidence',
        values='avg_lift'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_avg_lift, annot=True, fmt='.2f', cmap='viridis', ax=ax)
    ax.set_title('Average Lift of Generated Rules', fontsize=14, fontweight='bold')
    ax.set_xlabel('Minimum Confidence', fontsize=12)
    ax.set_ylabel('Minimum Support', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('parameter_effects_lift.png', dpi=300, bbox_inches='tight')
    print("  Saved visualization: parameter_effects_lift.png")
    
    # Analysis of parameter effects
    print("\n" + "-"*80)
    print("PARAMETER EFFECTS ANALYSIS")
    print("-"*80)


def analyze_product_categories(rules_df):
    """Analyze product category associations (OBJECTIVE 1)"""
    print("\n" + "="*80)
    print("OBJECTIVE 1: PRODUCT CATEGORY ASSOCIATIONS")
    print("="*80)
    
    # Filter for product-only rules
    product_rules = rules_df[rules_df['category'] == 'product_only'].copy()
    product_rules = product_rules.sort_values('lift', ascending=False)
    
    print(f"\nFound {len(product_rules)} product-only association rules")
    
    # Save product rules
    product_rules[["antecedent_str", "consequent_str", "support", "confidence", "lift", "category"]].to_csv('rules_product_categories.csv', index=False)
    print(f" Saved product category rules to rules_product_categories.csv")
    
    # Display top rules by different metrics
    print("\n" + "-"*80)
    print("TOP PRODUCT CATEGORY RULES")
    print("-"*80)
    
    print("\n1. TOP 5 RULES BY LIFT (Strongest Associations):")
    
    for idx, row in product_rules.head(5).iterrows():
        print(f"\nRule {idx + 1}:")
        print(f"  IF customer buys: {row['antecedent_str']}")
        print(f"  THEN they also buy: {row['consequent_str']}")
        print(f"  Support: {row['support']:.4f} ({row['support']*100:.2f}% of transactions)")
        print(f"  Confidence: {row['confidence']:.4f} ({row['confidence']*100:.1f}% probability)")
        print(f"  Lift: {row['lift']:.2f}x (products bought together {row['lift']:.2f}x more than by chance)")
    
    print("\n" + "-"*80)
    print("2. TOP 5 RULES BY SUPPORT (Most Frequent Patterns):")
    print("-"*80)
    
    top_support = product_rules.nlargest(5, 'support')
    for idx, row in top_support.iterrows():
        print(f"\nRule:")
        print(f"  {row['antecedent_str']} → {row['consequent_str']}")
        print(f"  Support: {row['support']:.4f}, Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.2f}")
    
    print("\n" + "-"*80)
    print("3. TOP 5 RULES BY CONFIDENCE (Most Reliable Predictions):")
    print("-"*80)
    
    top_confidence = product_rules.nlargest(5, 'confidence')
    for idx, row in top_confidence.iterrows():
        print(f"\nRule:")
        print(f"  {row['antecedent_str']} → {row['consequent_str']}")
        print(f"  Support: {row['support']:.4f}, Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.2f}")
    
    return product_rules


def analyze_purchase_timing(rules_df):
    """Analyze purchase timing patterns (OBJECTIVE 2)"""
    print("\n" + "="*80)
    print("OBJECTIVE 2: PURCHASE TIMING PATTERNS")
    print("="*80)
    
    # Filter for temporal-product rules
    temporal_rules = rules_df[rules_df['category'] == 'temporal_product'].copy()
    temporal_rules = temporal_rules.sort_values('lift', ascending=False)
    
    print(f"\nFound {len(temporal_rules)} temporal-product association rules")
    
    # Save temporal rules
    temporal_rules[["antecedent_str", "consequent_str", "support", "confidence", "lift", "category"]].to_csv('rules_purchase_timing.csv', index=False)
    print(f" Saved temporal rules to rules_purchase_timing.csv")
    
    # Display top temporal rules
    print("\n" + "-"*80)
    print("TOP PURCHASE TIMING RULES")
    print("-"*80)
    
    print("\n1. TOP 10 RULES BY LIFT (Strongest Time-Product Associations):")
    
    for idx, row in temporal_rules.head(10).iterrows():
        print(f"\nRule {idx + 1}:")
        print(f"  IF: {row['antecedent_str']}")
        print(f"  THEN: {row['consequent_str']}")
        print(f"  Support: {row['support']:.4f}, Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.2f}")
    
    print("\n" + "-"*80)
    print("2. DAY-OF-WEEK PATTERNS:")
    print("-"*80)
    
    dow_rules = temporal_rules[temporal_rules['antecedent_str'].str.contains('dow_|day_')]
    for idx, row in dow_rules.head(5).iterrows():
        print(f"\n  {row['antecedent_str']} → {row['consequent_str']}")
        print(f"  Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.2f}")
    
    print("\n" + "-"*80)
    print("3. TIME-OF-DAY PATTERNS:")
    print("-"*80)
    
    time_rules = temporal_rules[temporal_rules['antecedent_str'].str.contains('time_')]
    for idx, row in time_rules.head(5).iterrows():
        print(f"\n  {row['antecedent_str']} → {row['consequent_str']}")
        print(f"  Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.2f}")
    
    return temporal_rules


def visualize_rules(product_rules, temporal_rules):
    """Create visualizations for product and temporal rules"""
    print("\n" + "="*80)
    print("STEP 2.3: VISUALIZING TOP RULES")
    print("="*80)
    
    # Plot 1: Top product rules
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Product rules scatter plot
    top_prod = product_rules.head(20)
    
    temp_lift_normalized = (top_prod['lift'] - top_prod['lift'].min()) / (top_prod['lift'].max() - top_prod['lift'].min())
    temp_lift_scaled = temp_lift_normalized * 0.9 + 0.1     # normalize, scale top_prod['lift'] to values from 0.1 to 1
    
    axes[0].scatter(top_prod['support'], top_prod['confidence'], 
                   s=temp_lift_scaled*200+50, alpha=0.6, c=temp_lift_scaled, cmap='coolwarm')
    
    axes[0].set_xlabel('Support', fontsize=12)
    axes[0].set_ylabel('Confidence', fontsize=12)
    axes[0].set_title('Top 20 Product Category Rules (bubble size = lift)', 
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Add colorbar
    scatter = axes[0].scatter(top_prod['support'], top_prod['confidence'], 
                             s=temp_lift_scaled*200+50, alpha=0.6, c=temp_lift_scaled, 
                             cmap='coolwarm')
    plt.colorbar(scatter, ax=axes[0], label='Lift')
    
    # Temporal rules scatter plot
    top_temp = temporal_rules.head(20)
    
    axes[1].scatter(top_temp['support'], top_temp['confidence'], 
                   s=temp_lift_scaled*200+50, alpha=0.6, c=temp_lift_scaled, cmap='viridis')
    axes[1].set_xlabel('Support', fontsize=12)
    axes[1].set_ylabel('Confidence', fontsize=12)
    axes[1].set_title('Top 20 Temporal-Product Rules (bubble size = lift)', 
                      fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    scatter2 = axes[1].scatter(top_temp['support'], top_temp['confidence'], 
                              s=temp_lift_scaled*200+50, alpha=0.6, c=temp_lift_scaled, 
                              cmap='viridis')
    plt.colorbar(scatter2, ax=axes[1], label='Lift')
    
    plt.tight_layout()
    plt.savefig('top_rules_visualization.png', dpi=300, bbox_inches='tight')
    print("  Saved visualization: top_rules_visualization.png")


def generate_and_analyze_rules(transactions, min_support=0.04, min_confidence=0.2, min_lift=1.2):
    """Generate association rules and perform detailed analysis."""
    print("\n" + "="*80)
    print("STEP 3.1: GENERATING RULES WITH BALANCED PARAMETERS")
    print("="*80)
    
    print(f"\nUsing parameters:")
    print(f"  Minimum Support: {min_support}")
    print(f"  Minimum Confidence: {min_confidence}")
    print(f"  Minimum Lift: {min_lift}")
    
    print("\nRunning Apriori algorithm...")
    rules = list(apriori(
        transactions,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        min_length=2,
        max_length=10
    ))
    
    print(f"  Generated {len(rules)} association rules")
    
    print("\n" + "="*80)
    print("STEP 3.2: PARSING AND ORGANIZING RULES")
    print("="*80)
    
    # Parse rules into structured format
    parsed_rules = []
    
    for rule in rules:
        items = list(rule.items)
        for stat in rule.ordered_statistics:
            antecedent = list(stat.items_base)
            consequent = list(stat.items_add)
            
            parsed_rules.append({
                'antecedent': antecedent,
                'consequent': consequent,
                'antecedent_str': ', '.join(antecedent),
                'consequent_str': ', '.join(consequent),
                'support': rule.support,
                'confidence': stat.confidence,
                'lift': stat.lift
            })
    
    rules_df = pd.DataFrame(parsed_rules)
    print(f"  Parsed {len(rules_df)} rules")
    
    # Categorize rules
    def categorize_rule(row):
        """Categorize rule based on items involved"""
        
        all_items = row['antecedent'] + row['consequent']
        
        # Check for temporal items
        has_temporal = any(item.startswith(('dow_', 'time_', 'day_')) 
                          for item in all_items)
        
        # Check for product items (aisles)
        has_products = any(not item.startswith(('dow_', 'time_', 'day_')) 
                          for item in all_items)
        
        if has_temporal and has_products:
            return 'temporal_product'
        elif has_temporal:
            return 'temporal_only'
        elif has_products:
            return 'product_only'
        else:
            return 'other'
    
    rules_df['category'] = rules_df.apply(categorize_rule, axis=1)
    
    print(f"\nRule categories:")
    print(rules_df['category'].value_counts())
    
    # Save all rules
    rules_df[["antecedent_str", "consequent_str", "support", "confidence", "lift", "category"]].to_csv('association_rules_all.csv', index=False)
    print(f"\n  Saved all rules to association_rules_all.csv")
    
    # Analyze product categories (OBJECTIVE 1)
    product_rules = analyze_product_categories(rules_df)
    
    # Analyze purchase timing (OBJECTIVE 2)
    temporal_rules = analyze_purchase_timing(rules_df)
    
    # Create visualizations
    visualize_rules(product_rules, temporal_rules)

def run(file_path='transactions.pkl', min_support=0.04, min_confidence=0.2, min_lift=1.2, do_parameter_experimentation=False):
    """Main function to run the complete association rule mining pipeline."""
    # Step 1: Load data
    transactions = load_transactions(file_path)
    
    # Step 2: Parameter experimentation and visualization
    if(do_parameter_experimentation):
        parameter_experimentation(transactions, support_values=[0.04, 0.08, 0.15], confidence_values=[0.2, 0.4, 0.6])
    
    # Step 3: Generate and analyze rules
    generate_and_analyze_rules(transactions, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift)


if __name__ == "__main__":
    run()
