#!/usr/bin/env python3
"""
Test script to verify action server functionality
"""

import pandas as pd
import os

def test_search_functionality():
    """Test the same search logic used in the action server"""
    
    # Load the CSV data
    csv_path = "mitre_mitigations.csv"
    if not os.path.exists(csv_path):
        print("❌ CSV file not found")
        return
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} mitigations")
    
    # Test queries
    test_queries = [
        "T1174",
        "network segmentation", 
        "vulnerability scanning",
        "privilege escalation"
    ]
    
    print("\n🔍 Testing Search Logic:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Same logic as action server
        if any(char.isupper() and char.isdigit() for char in query):
            print(f"  → Searching by ID pattern: {query.upper()}")
            result = df[df['ID'].str.contains(query.upper(), na=False)]
            print(f"  → ID search results: {len(result)} matches")
        else:
            print(f"  → Searching by name/description: {query}")
            name_match = df[df['Name'].str.lower().str.contains(query, na=False)]
            desc_match = df[df['Description'].str.lower().str.contains(query, na=False)]
            result = pd.concat([name_match, desc_match]).drop_duplicates()
            print(f"  → Name matches: {len(name_match)}, Description matches: {len(desc_match)}")
            print(f"  → Total combined results: {len(result)}")
        
        if not result.empty:
            print(f"  ✅ Found {len(result)} results")
            for idx, row in result.head(2).iterrows():
                print(f"    • {row['ID']}: {row['Name']}")
        else:
            print(f"  ❌ No results found")

if __name__ == "__main__":
    test_search_functionality()
