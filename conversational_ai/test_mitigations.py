#!/usr/bin/env python3
"""
Test script to demonstrate MITRE ATT&CK mitigation lookup functionality
"""

import pandas as pd
import os

def test_mitigation_lookup():
    """Test the mitigation lookup functionality"""
    
    # Load the CSV data
    csv_path = "mitre_mitigations.csv"
    if not os.path.exists(csv_path):
        print("❌ CSV file not found. Please run extract_mitigations.py first.")
        return
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} mitigations from {csv_path}")
    
    # Test queries
    test_queries = [
        "vulnerability scanning",
        "M1016",
        "privilege escalation",
        "network segmentation",
        "ransomware",
        "lateral movement"
    ]
    
    print("\n🔍 Testing Mitigation Lookup:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Search by ID (e.g., M1016)
        if any(char.isupper() and char.isdigit() for char in query):
            result = df[df['ID'].str.contains(query.upper(), na=False)]
        else:
            # Search by name or description
            name_match = df[df['Name'].str.lower().str.contains(query, na=False)]
            desc_match = df[df['Description'].str.lower().str.contains(query, na=False)]
            result = pd.concat([name_match, desc_match]).drop_duplicates()
        
        if not result.empty:
            if len(result) == 1:
                row = result.iloc[0]
                print(f"✅ Found: {row['ID']} - {row['Name']}")
                print(f"   Description: {row['Description'][:100]}...")
            else:
                print(f"🔍 Found {len(result)} matches:")
                for idx, row in result.head(3).iterrows():
                    print(f"   • {row['ID']}: {row['Name']}")
                if len(result) > 3:
                    print(f"   ... and {len(result) - 3} more")
        else:
            print("❌ No matches found")
    
    # Show sample of available mitigations
    print("\n📋 Sample Available Mitigations:")
    print("=" * 50)
    sample = df.head(10)
    for idx, row in sample.iterrows():
        print(f"• {row['ID']}: {row['Name']}")

if __name__ == "__main__":
    test_mitigation_lookup()
