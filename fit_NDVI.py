#!/usr/bin/env python3
"""
Script to fit NDVI inference data into LST time steps.

For each NDVI inference image in the 'ndvi_infer_8days' directory, finds the closest LST date
within a specified window (default ±4 days), renames the NDVI file to have the LST date,
and removes any NDVI files not matched to any LST date.
"""

import os
import re
import argparse
from datetime import datetime

def parse_date_from_filename(filename):
    """
    Extracts a date in YYYYMMDD or YYYY-MM-DD format from the filename.
    Returns a datetime.date or None if no valid date is found.
    """
    # Try matching YYYY-MM-DD first
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass
    # Fallback to YYYYMMDD
    match = re.search(r"(\d{8})", filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            pass
    return None


def find_lst_dates(lst_dir):
    """
    Lists all files in lst_dir, parses dates from filenames, and returns a sorted list of unique dates.
    """
    files = [f for f in os.listdir(lst_dir) if os.path.isfile(os.path.join(lst_dir, f))]
    dates = set()
    for f in files:
        date = parse_date_from_filename(f)
        if date:
            dates.add(date)
    return sorted(dates)


def find_ndvi_files(ndvi_dir):
    """
    Lists all files in ndvi_dir, parses dates from filenames, and returns a list of (date, filename).
    """
    files = [f for f in os.listdir(ndvi_dir) if os.path.isfile(os.path.join(ndvi_dir, f))]
    date_files = []
    for f in files:
        date = parse_date_from_filename(f)
        if date:
            date_files.append((date, f))
    return date_files


def fit_ndvi_to_lst(lst_dates, ndvi_date_files, window_days):
    """
    Matches each NDVI file to the closest LST date within window_days.
    Returns a dict mapping matched LST date -> (ndvi_filename, delta_days).
    """
    mapping = {}
    for ndvi_date, fname in ndvi_date_files:
        # Find LST dates within the window
        candidates = [(abs((lst_date - ndvi_date).days), lst_date) for lst_date in lst_dates
                      if abs((lst_date - ndvi_date).days) <= window_days]
        if not candidates:
            continue
        # Choose the candidate with minimum difference
        delta, best_lst_date = min(candidates, key=lambda x: x[0])
        existing = mapping.get(best_lst_date)
        if existing is None or delta < existing[1]:
            mapping[best_lst_date] = (fname, delta)
    return mapping


def main(data_dir, window_days):
    lst_dir = os.path.join(data_dir, 'lst')
    # ndvi_dir = os.path.join(data_dir, 'ndvi_infer')
    ndvi_dir = None
    for folder in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, folder)):
            if "ndvi8days" in folder:
                ndvi_dir = os.path.join(data_dir, folder)
                break
    

    if not os.path.isdir(lst_dir):
        print(f"Error: LST directory not found: {lst_dir}")
        return
    if not os.path.isdir(ndvi_dir):
        print(f"Error: NDVI directory not found: {ndvi_dir}")
        return

    lst_dates = find_lst_dates(lst_dir)
    if not lst_dates:
        print(f"No valid LST dates found in {lst_dir}")
        return

    ndvi_date_files = find_ndvi_files(ndvi_dir)
    if not ndvi_date_files:
        print(f"No valid NDVI files found in {ndvi_dir}")
        return

    mapping = fit_ndvi_to_lst(lst_dates, ndvi_date_files, window_days)
    if not mapping:
        print("No NDVI files matched to LST dates within the given window.")
    else:
        print(f"Matched {len(mapping)} NDVI files to LST dates.")

    # Rename matched NDVI files
    chosen_files = set()
    for lst_date, (ndvi_fname, delta) in mapping.items():
        orig_path = os.path.join(ndvi_dir, ndvi_fname)
        # Generate new date strings in hyphenated and compact formats
        new_date_str_hyph = lst_date.strftime('%Y-%m-%d')
        new_date_str_compact = lst_date.strftime('%Y%m%d')
        # Attempt to replace hyphenated date first
        new_fname = re.sub(r"\d{4}-\d{2}-\d{2}", new_date_str_hyph, ndvi_fname, count=1)
        if new_fname == ndvi_fname:
            # Fallback: replace compact YYYYMMDD pattern
            new_fname = re.sub(r"\d{8}", new_date_str_compact, ndvi_fname, count=1)
            if new_fname == ndvi_fname:
                # As a last resort, prefix with hyphenated date
                new_fname = f"{new_date_str_hyph}_{ndvi_fname}"
        new_path = os.path.join(ndvi_dir, new_fname)
        try:
            os.rename(orig_path, new_path)
            print(f"Renamed '{ndvi_fname}' to '{new_fname}' (delta {delta} days)")
            chosen_files.add(new_fname)
        except Exception as e:
            print(f"Warning: Failed to rename '{orig_path}' to '{new_path}': {e}")

    # Remove unmatched files
    all_ndvi_files = [f for f in os.listdir(ndvi_dir) if os.path.isfile(os.path.join(ndvi_dir, f))]
    for fname in all_ndvi_files:
        if fname not in chosen_files:
            try:
                os.remove(os.path.join(ndvi_dir, fname))
                print(f"Removed unmatched NDVI file: {fname}")
            except Exception as e:
                print(f"Warning: Failed to remove '{fname}': {e}")

    print("NDVI fitting to LST timesteps completed.")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description="Fit NDVI inference files to LST time steps by renaming and cleanup"
    # )
    # parser.add_argument(
    #     '--data-dir',
    #     required=True,
    #     help="Path to the data directory containing 'lst' and 'ndvi_infer_8days' subdirectories"
    # )
    # parser.add_argument(
    #     '--window',
    #     type=int,
    #     default=4,
    #     help="Window (in days) around NDVI dates to search for matching LST dates"
    # )
    # args = parser.parse_args()
    # main(args.data_dir, args.window)
    
    data_dir = "/mnt/hdd12tb/code/nhatvm/DELAG/data_grid_base"
    for roi in os.listdir(data_dir):
        if roi == 'output_models':
            continue
        lst_dir = os.path.join(data_dir, roi, 'lst')
        ndvi_dir = None
        for folder in os.listdir(os.path.join(data_dir, roi)):
            if os.path.isdir(os.path.join(data_dir, roi, folder)):
                if "ndvi8days" in folder:
                    ndvi_dir = os.path.join(data_dir, roi, folder)
                    break
        if not os.path.isdir(lst_dir):
            print(f"Error: LST directory not found: {lst_dir}")
            continue
        if not os.path.isdir(ndvi_dir):
            print(f"Error: NDVI directory not found: {ndvi_dir}")
            continue
        main(os.path.join(data_dir, roi), 4)
        
