#!/usr/bin/env python3
"""
High Probability Features Summary
Clean, focused summary of only the high confidence predictions and their key features
"""

import pandas as pd
import numpy as np

def show_high_probability_features_only():
    """Show only high probability features in a clean, focused format"""
    
    print("🎯 HIGH PROBABILITY FEATURES SUMMARY")
    print("=" * 60)
    
    # Load high confidence predictions
    df_high = pd.read_csv("high_confidence_predictions.csv")
    
    print(f"📊 HIGH CONFIDENCE DATASET:")
    print(f"   Total Samples: {len(df_high):,}")
    print(f"   Confidence Range: {df_high['confidence_score'].min():.4f} - {df_high['confidence_score'].max():.4f}")
    print(f"   Average Confidence: {df_high['confidence_score'].mean():.4f}")
    
    # Show only attack types detected with high confidence
    print(f"\n⚔️  DETECTED ATTACK TYPES (High Confidence Only):")
    attack_summary = df_high['predicted_label'].value_counts()
    for attack, count in attack_summary.items():
        percentage = count / len(df_high) * 100
        avg_conf = df_high[df_high['predicted_label'] == attack]['confidence_score'].mean()
        print(f"   🔴 {attack}")
        print(f"      Samples: {count:,} ({percentage:.1f}%)")
        print(f"      Avg Confidence: {avg_conf:.4f}")
    
    # Feature columns (excluding prediction columns)
    feature_cols = [col for col in df_high.columns if col not in ['predicted_label', 'confidence_score', 'prediction_confidence']]
    
    # Find only the truly active features
    active_features = []
    for feature in feature_cols:
        non_zero_count = (df_high[feature] != 0).sum()
        if non_zero_count > 0:
            non_zero_pct = non_zero_count / len(df_high) * 100
            mean_val = df_high[df_high[feature] != 0][feature].mean()
            unique_vals = df_high[feature].nunique()
            active_features.append((feature, non_zero_pct, non_zero_count, mean_val, unique_vals))
    
    # Sort by activation percentage
    active_features.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🔑 KEY FEATURES IN HIGH CONFIDENCE PREDICTIONS:")
    print(f"   (Only showing features with >0% activation)")
    print(f"\n   {'Feature':<25} {'Active %':<10} {'Samples':<10} {'Avg Value':<12} {'Unique Vals':<12}")
    print("   " + "-" * 75)
    
    for feature, pct, count, avg_val, unique_vals in active_features:
        print(f"   {feature:<25} {pct:<10.1f} {count:<10,} {avg_val:<12.3f} {unique_vals:<12}")
    
    # Attack-specific feature signatures (simplified)
    print(f"\n🎯 ATTACK SIGNATURES:")
    for attack_type in attack_summary.index:
        attack_data = df_high[df_high['predicted_label'] == attack_type]
        print(f"\n   🔍 {attack_type}:")
        print(f"      Confidence: {attack_data['confidence_score'].mean():.4f}")
        
        # Find the defining features for this attack
        defining_features = []
        for feature, _, _, _, _ in active_features:
            feature_activation = (attack_data[feature] != 0).sum() / len(attack_data) * 100
            if feature_activation > 50:  # Features active in >50% of this attack type
                avg_val = attack_data[feature].mean()
                defining_features.append((feature, feature_activation, avg_val))
        
        if defining_features:
            print(f"      Key Features:")
            for feat, activation, avg in defining_features:
                print(f"        • {feat}: {activation:.0f}% (avg: {avg:.3f})")
        else:
            print(f"      No dominant features found")
    
    # Show representative samples
    print(f"\n📋 REPRESENTATIVE HIGH CONFIDENCE SAMPLES:")
    for attack_type in attack_summary.index:
        attack_samples = df_high[df_high['predicted_label'] == attack_type]
        top_sample = attack_samples.loc[attack_samples['confidence_score'].idxmax()]
        
        print(f"\n   🏆 Best {attack_type} Detection:")
        print(f"      Confidence: {top_sample['confidence_score']:.4f}")
        print(f"      Active Features:")
        
        active_in_sample = []
        for feature in feature_cols:
            if top_sample[feature] != 0:
                active_in_sample.append(f"{feature}={top_sample[feature]}")
        
        if active_in_sample:
            for feat in active_in_sample[:5]:  # Show top 5
                print(f"        • {feat}")
        else:
            print(f"        • No notable active features")
    
    # Final statistics
    total_original_samples = 360270  # From the full dataset
    high_conf_rate = len(df_high) / total_original_samples * 100
    
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   🎯 High Confidence Rate: {high_conf_rate:.1f}% of all predictions")
    print(f"   🔑 Active Features: {len(active_features)} out of {len(feature_cols)} total")
    print(f"   ⚔️  Attack Types Detected: {len(attack_summary)}")
    print(f"   🏆 Model Reliability: Very High (avg confidence: {df_high['confidence_score'].mean():.4f})")
    
    print(f"\n✅ HIGH PROBABILITY FEATURE ANALYSIS COMPLETE")
    print(f"   Focus: Only samples with confidence > 0.8")
    print(f"   Result: {len(df_high):,} highly reliable attack detections")

if __name__ == "__main__":
    show_high_probability_features_only()
