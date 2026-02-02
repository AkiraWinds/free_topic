"""
Calculate average LLM-extracted nodes and relationships from a folder of JSON files.

Usage:
    python3 data_analysis.py
"""

import os
import json
from pathlib import Path

def calculate_stats(folder_path: str):
    """
    Calculate statistics from all JSON files in the folder
    
    Args:
        folder_path: Path to folder containing JSON files
    
    Returns:
        dict: Statistics including totals and averages
    """
    total_nodes = 0
    total_relationships = 0
    file_count = 0
    
    # Track individual file stats for detailed output
    file_stats = []
    
    # Iterate through all JSON files
    for filename in os.listdir(folder_path):
        if not filename.endswith('.json'):
            continue
            
        file_path = os.path.join(folder_path, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract graph stats
            graph_stats = data.get('meta', {}).get('graph_stats', {})
            llm_nodes = graph_stats.get('llm_extracted_nodes', 0)
            llm_rels = graph_stats.get('llm_extracted_relationships', 0)
            
            # Accumulate totals
            total_nodes += llm_nodes
            total_relationships += llm_rels
            file_count += 1
            
            # Store individual file stats
            file_stats.append({
                'filename': filename,
                'nodes': llm_nodes,
                'relationships': llm_rels
            })
            
        except Exception as e:
            print(f"⚠ Error reading {filename}: {e}")
            continue
    
    # Calculate averages
    avg_nodes = total_nodes / file_count if file_count > 0 else 0
    avg_relationships = total_relationships / file_count if file_count > 0 else 0
    
    return {
        'file_count': file_count,
        'total_nodes': total_nodes,
        'total_relationships': total_relationships,
        'avg_nodes': avg_nodes,
        'avg_relationships': avg_relationships,
        'file_stats': file_stats
    }


def main():
    # Configure folder path
    folder_path = "../../../data/neo4j_data/Legal_Discourse_Graph/ollama/act-1997-078/"
    
    print(f"Analyzing JSON files in: {folder_path}")
    print("="*60)
    
    # Calculate statistics
    stats = calculate_stats(folder_path)
    
    # Display results
    print(f"\n📊 Summary Statistics:")
    print(f"{'='*60}")
    print(f"Total JSON files processed: {stats['file_count']}")
    print(f"\n🔢 Nodes:")
    print(f"  Total llm_extracted_nodes: {stats['total_nodes']}")
    print(f"  Average llm_extracted_nodes: {stats['avg_nodes']:.2f}")
    print(f"\n🔗 Relationships:")
    print(f"  Total llm_extracted_relationships: {stats['total_relationships']}")
    print(f"  Average llm_extracted_relationships: {stats['avg_relationships']:.2f}")
    print(f"{'='*60}")
    
    # Display detailed breakdown (optional)
    print(f"\n📋 Detailed Breakdown:")
    print(f"{'Filename':<60} {'Nodes':>8} {'Rels':>8}")
    print("-"*80)
    for file_stat in stats['file_stats']:
        print(f"{file_stat['filename']:<60} {file_stat['nodes']:>8} {file_stat['relationships']:>8}")
    
    # Save results to JSON
    output_file = os.path.join(folder_path, "_statistics.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Statistics saved to: {output_file}")


if __name__ == "__main__":
    main()