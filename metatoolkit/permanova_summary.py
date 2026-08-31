#!/usr/bin/env python
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Compile and format PERMANOVA results.")
    parser.add_argument('-i', '--input', required=True, help="Input PERMANOVA TSV")
    parser.add_argument('-o', '--output', required=True, help="Output compiled TSV")
    args = parser.parse_args()

    # Load the data
    df = pd.read_csv(args.input, sep='\t')

    # 1. Extract Metric and Model Type
    #df['Metric'] = df['metrics'].str.extract(r'adonisoutput_(?:un)?strata/([^/]+)/')
    df['Metric'] = df['metrics'].str.extract(r'/([^/]+)/permanova_marginal')
    df['Model_Source'] = df['metrics'].apply(
        lambda x: 'Stratified' if 'strata/' in x and 'unstrata' not in x else 'Unstratified'
    )
    df[['Asian_Ethnicity', 'Intervention']] = df['metrics'].str.extract(r'([A-Za-z]+)_ethnicity_([A-Za-z]+)')

    # 2. Cleanup
    df = df.loc[~df['Coefficient'].isin(['Residual', 'Total'])].copy()
    df['Coefficient'] = df['Coefficient'].str.replace('`', '', regex=False).str.replace('TRUE', '', regex=False)

    # 3. Selection Logic
    is_strat_appropriate = df['Coefficient'].str.contains('timepoint|:', regex=True)
    mask = (is_strat_appropriate & (df['Model_Source'] == 'Stratified')) | \
           (~is_strat_appropriate & (df['Model_Source'] == 'Unstratified'))
    #df_final = df[mask].copy()
    df_final = df.copy()

    # 4. Final Column Organizing
    df_final = df_final.rename(columns={'Pr(>F)': 'p_value', 'Model_Source': 'Model'})
    cols = ['Metric', 'Coefficient', 'Df', 'R2', 'F', 'p_value', 'Asian_Ethnicity', 'Intervention']
    df_final = df_final[cols].sort_values(['Metric', 'p_value'])

    # Formatting
    df_final['R2'] = df_final['R2'].round(5)
    df_final['F'] = df_final['F'].round(3)

    # Save
    df_final.to_csv(args.output, sep='\t', index=False)
    print(f"Successfully compiled PERMANOVA results to {args.output}")

if __name__ == "__main__":
    main()
