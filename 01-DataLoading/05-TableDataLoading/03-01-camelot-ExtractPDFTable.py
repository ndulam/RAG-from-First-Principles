import camelot
import pandas as pd
# from ctypes.util import find_library
# find_library("gs")

pdf_path = "90-Data/ComplexPDF/billionaires_page-1-5.pdf"
import time

start_time = time.time()
tables = camelot.read_pdf(pdf_path, pages="all")
end_time = time.time()
print(f"PDF table parsing took: {end_time - start_time:.2f} seconds")

# Convert every table to a DataFrame
if tables:
    # Iterate over all the tables
    for i, table in enumerate(tables, 1):
        # Convert the table to a DataFrame
        df = table.df

        # Print the current table's data
        print(f"\nTable {i} data:")
        print(df)

        # Show basic info
        print(f"\nTable {i} basic info:")
        print(df.info())

        # Save to a CSV file
        csv_filename = f"billionaires_table_{i}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\nTable {i} data saved to {csv_filename}")
