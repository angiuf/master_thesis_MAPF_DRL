#!/usr/bin/env python3
"""
Script to update SILLM time field with the formula: new_time = time / 256 * episode_length
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import shutil
from datetime import datetime

def find_sillm_csv_files(base_path):
    """Find all SILLM CSV files in the final_data directory"""
    pattern = str(base_path / "final_data" / "*" / "SILLM" / "*.csv")
    csv_files = glob.glob(pattern)
    return [Path(f) for f in csv_files]

def backup_file(file_path):
    """Create a backup of the original file"""
    backup_path = file_path.with_suffix(f'.csv.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    return backup_path

def update_time_field(df):
    """Update the time field using the formula: new_time = time / 256 * episode_length"""
    if 'time' not in df.columns:
        print("Warning: 'time' column not found in dataframe")
        return df, False
    
    if 'episode_length' not in df.columns:
        print("Warning: 'episode_length' column not found in dataframe")
        return df, False
    
    # Store original time values for comparison
    original_time = df['time'].copy()
    
    # Apply the formula: new_time = time / 256 * episode_length
    df['time'] = df['time'] / 256 * df['episode_length']
    
    # Also update time-related statistics if they exist
    time_stats_columns = [col for col in df.columns if col.startswith('time_') and col != 'time']
    
    for col in time_stats_columns:
        if col in df.columns:
            # Apply the same transformation to statistics
            df[col] = df[col] / 256 * df['episode_length']
    
    print(f"Updated time field and {len(time_stats_columns)} time statistics columns")
    print(f"Sample transformation: {original_time.iloc[0]:.6f} -> {df['time'].iloc[0]:.6f}")
    
    return df, True

def process_sillm_file(file_path, create_backup=True):
    """Process a single SILLM CSV file"""
    print(f"\nProcessing: {file_path}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Show some info about the data
        if 'n_agents' in df.columns:
            print(f"Agent counts: {sorted(df['n_agents'].unique())}")
        
        # Create backup if requested
        if create_backup:
            backup_path = backup_file(file_path)
        
        # Update time field
        updated_df, success = update_time_field(df)
        
        if success:
            # Save the updated file
            updated_df.to_csv(file_path, index=False)
            print(f"Successfully updated and saved: {file_path}")
            return True
        else:
            print(f"Failed to update: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to process all SILLM files"""
    base_path = Path("/home/andrea/CODE/master_thesis_MAPF_DRL/results")
    
    print("SILLM Time Field Update Script")
    print("="*50)
    print(f"Formula: new_time = time / 256 * episode_length")
    print(f"Base path: {base_path}")
    
    # Find all SILLM CSV files
    sillm_files = find_sillm_csv_files(base_path)
    
    if not sillm_files:
        print("No SILLM CSV files found!")
        return
    
    print(f"\nFound {len(sillm_files)} SILLM CSV files:")
    for file_path in sillm_files:
        print(f"  {file_path}")
    
    # Ask for confirmation
    response = input(f"\nProceed with updating {len(sillm_files)} files? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Aborted by user.")
        return
    
    # Process each file
    successful = 0
    failed = 0
    
    for file_path in sillm_files:
        if process_sillm_file(file_path, create_backup=True):
            successful += 1
        else:
            failed += 1
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Successfully updated: {successful} files")
    print(f"Failed: {failed} files")
    
    if successful > 0:
        print(f"\nBackup files were created with timestamp suffixes.")
        print(f"Original files have been updated with the new time values.")

if __name__ == "__main__":
    main()
