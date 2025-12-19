"""
ChEMBL data acquisition module.

Downloads bioactivity data from ChEMBL and converts units to standardised format.
"""

import pandas as pd


def download_chembl_target(target_info, activity_type='IC50', max_compounds=1000):
    """
    Download data for a single target from ChEMBL.

    Args:
        target_info (dict): Dict with 'chembl_id' and 'name' keys
        activity_type (str): Activity type to download (IC50, Ki, EC50)
        max_compounds (int): Maximum number of compounds to download

    Returns:
        pd.DataFrame: Bioactivity data or None if download fails
    """
    chembl_id = target_info['chembl_id']
    name = target_info['name']

    try:
        from chembl_webresource_client.new_client import new_client

        print(f"\n  Downloading {name} ({chembl_id})...")

        activity = new_client.activity

        activities = activity.filter(
            target_chembl_id=chembl_id,
            type=activity_type,
            relation='=',
            assay_type='B'
        ).only(
            'molecule_chembl_id',
            'canonical_smiles',
            'standard_value',
            'standard_units',
            'standard_type',
            'standard_relation'
        )

        results = []
        count = 0

        for record in activities:
            results.append(record)
            count += 1
            if count >= max_compounds:
                break

        if len(results) == 0:
            print(f"    ⚠️ No data for {name}")
            return None

        df = pd.DataFrame(results)

        # Filter
        df = df[df['standard_type'] == activity_type]
        df = df[df['standard_relation'] == '=']

        # Convert units
        df = convert_units_to_nM(df)

        # Add target info
        df['target_name'] = name
        df['target_chembl_id'] = chembl_id

        print(f"    ✓ {len(df)} records")
        return df

    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return None


def convert_units_to_nM(df):
    """
    Convert activity values to nM.

    Args:
        df (pd.DataFrame): DataFrame with 'standard_value' and 'standard_units' columns

    Returns:
        pd.DataFrame: DataFrame with values converted to nM
    """
    df = df.copy()

    for idx, row in df.iterrows():
        value = row['standard_value']
        unit = row['standard_units']

        if pd.isna(value) or pd.isna(unit):
            continue

        if unit == 'uM':
            df.at[idx, 'standard_value'] = value * 1000
            df.at[idx, 'standard_units'] = 'nM'
        elif unit == 'pM':
            df.at[idx, 'standard_value'] = value / 1000
            df.at[idx, 'standard_units'] = 'nM'
        elif unit == 'M':
            df.at[idx, 'standard_value'] = value * 1e9
            df.at[idx, 'standard_units'] = 'nM'

    df = df[df['standard_units'] == 'nM']
    return df


def load_user_csv_multitarget(csv_path):
    """
    Load user CSV with multiple targets.

    Args:
        csv_path (str): Path to CSV file

    Returns:
        pd.DataFrame: Loaded and standardised data

    Expected CSV columns:
        - smiles (required)
        - activity_value (required)
        - target_name (required)
        - activity_unit (optional, defaults to nM)
        - activity_type (optional, defaults to IC50)
    """
    print(f"\nLoading user CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # Required columns
    if 'smiles' not in df.columns or 'activity_value' not in df.columns or 'target_name' not in df.columns:
        raise ValueError("CSV must have: smiles, activity_value, target_name")

    # Rename
    df = df.rename(columns={
        'smiles': 'canonical_smiles',
        'activity_value': 'standard_value'
    })

    # Defaults
    if 'activity_unit' not in df.columns:
        df['standard_units'] = 'nM'
    else:
        df['standard_units'] = df['activity_unit']

    df['standard_type'] = df.get('activity_type', 'IC50')
    df['standard_relation'] = df.get('relation', '=')

    # Filter
    df = df[df['standard_relation'] == '=']
    df = convert_units_to_nM(df)

    print(f"✓ Loaded {len(df)} records")
    return df
