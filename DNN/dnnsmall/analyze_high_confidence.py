#!/usr/bin/env python3
"""
High Probability Features Detailed Analysis
Analyzes the patterns and characteristics of high confidence predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_high_probability_features():
    """Detailed analysis of high probability features"""
    
    print("🎯 HIGH PROBABILITY FEATURES DETAILED ANALYSIS")
    print("=" * 80)
    
    # Load high confidence predictions
    df_high = pd.read_csv("high_confidence_predictions.csv")
    
    print(f"📊 Dataset Overview:")
    print(f"   Total High Confidence Samples: {len(df_high):,}")
    print(f"   Confidence Threshold: >0.8")
    print(f"   Average Confidence: {df_high['confidence_score'].mean():.4f}")
    print(f"   Min Confidence: {df_high['confidence_score'].min():.4f}")
    print(f"   Max Confidence: {df_high['confidence_score'].max():.4f}")
    
    # Attack type distribution
    print(f"\n🎯 ATTACK TYPE DISTRIBUTION:")
    attack_dist = df_high['predicted_label'].value_counts()
    for attack, count in attack_dist.items():
        pct = count / len(df_high) * 100
        avg_conf = df_high[df_high['predicted_label'] == attack]['confidence_score'].mean()
        min_conf = df_high[df_high['predicted_label'] == attack]['confidence_score'].min()
        max_conf = df_high[df_high['predicted_label'] == attack]['confidence_score'].max()
        print(f"   {attack}:")
        print(f"     Count: {count:,} ({pct:.1f}%)")
        print(f"     Confidence: {avg_conf:.4f} ± {df_high[df_high['predicted_label'] == attack]['confidence_score'].std():.4f}")
        print(f"     Range: [{min_conf:.4f}, {max_conf:.4f}]")
    
    # Feature columns (excluding prediction columns)
    feature_cols = [col for col in df_high.columns if col not in ['predicted_label', 'confidence_score', 'prediction_confidence']]
    
    print(f"\n🔬 DETAILED FEATURE ANALYSIS:")
    print(f"   Total Features: {len(feature_cols)}")
    
    # Calculate feature importance based on activation
    feature_importance = {}
    for feature in feature_cols:
        # Count non-zero values
        non_zero_count = (df_high[feature] != 0).sum()
        non_zero_pct = non_zero_count / len(df_high) * 100
        
        # Calculate statistics for non-zero values
        non_zero_values = df_high[df_high[feature] != 0][feature]
        if len(non_zero_values) > 0:
            mean_val = non_zero_values.mean()
            std_val = non_zero_values.std()
            max_val = non_zero_values.max()
            min_val = non_zero_values.min()
        else:
            mean_val = std_val = max_val = min_val = 0
        
        feature_importance[feature] = {
            'activation_pct': non_zero_pct,
            'count': non_zero_count,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val
        }
    
    # Sort by activation percentage
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1]['activation_pct'], reverse=True)
    
    print(f"\n📈 FEATURE ACTIVATION RANKING:")
    print(f"{'Rank':<5} {'Feature':<25} {'Active%':<8} {'Count':<8} {'Mean':<10} {'Std':<10} {'Range':<15}")
    print("-" * 90)
    
    for rank, (feature, stats) in enumerate(sorted_features, 1):
        if stats['activation_pct'] > 0:  # Only show active features
            range_str = f"[{stats['min']:.1f}, {stats['max']:.1f}]"
            print(f"{rank:<5} {feature:<25} {stats['activation_pct']:<8.1f} {stats['count']:<8} {stats['mean']:<10.3f} {stats['std']:<10.3f} {range_str:<15}")
    
    # Attack-specific feature signatures
    print(f"\n🔍 ATTACK-SPECIFIC FEATURE SIGNATURES:")
    for attack_type in attack_dist.index:
        print(f"\n⚔️  {attack_type} Attack Pattern:")
        attack_data = df_high[df_high['predicted_label'] == attack_type]
        
        # Find discriminative features for this attack
        discriminative_features = []
        for feature in feature_cols:
            activation_pct = (attack_data[feature] != 0).sum() / len(attack_data) * 100
            if activation_pct > 5:  # Features active in >5% of samples
                mean_val = attack_data[feature].mean()
                discriminative_features.append((feature, activation_pct, mean_val))
        
        # Sort by activation percentage
        discriminative_features.sort(key=lambda x: x[1], reverse=True)
        
        if discriminative_features:
            print(f"     Key Features:")
            for feature, pct, mean_val in discriminative_features[:10]:
                print(f"       {feature}: {pct:.1f}% active (avg: {mean_val:.3f})")
        else:
            print(f"     No highly discriminative features found")
    
    # Confidence distribution analysis
    print(f"\n📊 CONFIDENCE DISTRIBUTION ANALYSIS:")
    conf_bins = [0.8, 0.85, 0.9, 0.95, 1.0]
    conf_labels = ['0.8-0.85', '0.85-0.9', '0.9-0.95', '0.95-1.0']
    
    df_high['conf_bin'] = pd.cut(df_high['confidence_score'], bins=conf_bins, labels=conf_labels, include_lowest=True)
    conf_dist = df_high['conf_bin'].value_counts().sort_index()
    
    for bin_label, count in conf_dist.items():
        pct = count / len(df_high) * 100
        bin_data = df_high[df_high['conf_bin'] == bin_label]
        attack_breakdown = bin_data['predicted_label'].value_counts()
        print(f"   {bin_label}: {count:,} samples ({pct:.1f}%)")
        for attack, attack_count in attack_breakdown.head(3).items():
            attack_pct = attack_count / count * 100
            print(f"     └─ {attack}: {attack_count:,} ({attack_pct:.1f}%)")
    
    # Sample analysis for each attack type
    print(f"\n🔍 SAMPLE PATTERN ANALYSIS:")
    for attack_type in attack_dist.index[:2]:  # Top 2 attack types
        print(f"\n📋 {attack_type} Sample Patterns:")
        attack_samples = df_high[df_high['predicted_label'] == attack_type].head(3)
        
        for idx, (_, sample) in enumerate(attack_samples.iterrows(), 1):
            print(f"   Sample {idx} (Confidence: {sample['confidence_score']:.4f}):")
            active_features = []
            for feature in feature_cols:
                if sample[feature] != 0:
                    active_features.append(f"{feature}={sample[feature]}")
            
            if active_features:
                print(f"     Active: {', '.join(active_features[:5])}")
                if len(active_features) > 5:
                    print(f"     ... and {len(active_features)-5} more")
            else:
                print(f"     No active features")
    
    print(f"\n✅ SUMMARY:")
    print(f"   🎯 {len(df_high):,} high confidence predictions analyzed")
    print(f"   📊 {len([f for f in sorted_features if f[1]['activation_pct'] > 0])} active features identified")
    print(f"   ⚔️  {len(attack_dist)} attack types detected")
    print(f"   🏆 Average confidence: {df_high['confidence_score'].mean():.4f}")
    
    # Save detailed analysis
    analysis_summary = {
        'total_samples': len(df_high),
        'attack_distribution': attack_dist.to_dict(),
        'avg_confidence': df_high['confidence_score'].mean(),
        'feature_ranking': {f: stats for f, stats in sorted_features[:10]}
    }
    
    return analysis_summary

if __name__ == "__main__":
    summary = analyze_high_probability_features()
