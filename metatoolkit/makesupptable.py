#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, Border, Side

# Paths
supp_table_list_path = 'figures/suppTableList.txt'
contents_path = 'figures/contents.tsv'
glossary_path = 'figures/glossary.tsv'
output_path = 'figures/suppTables.xlsx'

# Read the list of tables to be included
tables = pd.read_csv(supp_table_list_path, header=None)
table_paths = 'results/' + tables[0] + '.tsv'

# Add Glossary
table_paths = pd.concat([pd.Series([contents_path, glossary_path]), table_paths])

# Write tables to Excel file
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for j, table_path in enumerate(table_paths):
        # Read each table and export to the Excel file
        df = pd.read_csv(table_path, sep='\t', index_col=0)
        sheet_name = f"Data_{j-1}"
        if j == 0:
            sheet_name = 'Contents'
        if j == 1:
            sheet_name = 'Glossary'
        df.to_excel(writer, sheet_name=sheet_name)

# Define border styles
thin_border = Border(bottom=Side(style="thin"))

# Apply formatting
workbook = load_workbook(output_path)
for sheet_name in workbook.sheetnames:
    sheet = workbook[sheet_name]

    # Clear formatting: remove bold in the first column and all borders
    for row in sheet.iter_rows():
        for cell in row:
            cell.font = Font(bold=False)  # Remove bold from all cells
            cell.border = Border()  # Remove all borders

    # Apply header formatting (bold, center, bottom border)
    for cell in sheet[1]:  # Assumes first row is the header
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')
        cell.border = thin_border  # Apply bottom border only to header

    # Set column width based on content
    for col in sheet.columns:
        max_length = max(len(str(cell.value)) for cell in col if cell.value) + 2
        col_idx = col[0].column if col[0].column is not None else 1
        col_letter = get_column_letter(col_idx)
        sheet.column_dimensions[col_letter].width = max(max_length, 12)

# Save formatted workbook
workbook.save(output_path)
