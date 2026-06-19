import pdfplumber
import pandas as pd
import time

# Record the start time
start_time = time.time()

# Open the PDF file
pdf = pdfplumber.open("90-Data/ComplexPDF/billionaires_page-1-5.pdf")

# Iterate over every page
for page in pdf.pages:
    # Extract tables
    tables = page.extract_tables()

    # Check whether any tables were found
    if tables:
        print(f"Found {len(tables)} table(s) on page {page.page_number}")

        # Iterate over all the tables on this page
        for i, table in enumerate(tables):
            print(f"\nProcessing table {i+1}:")

            # Convert the table to a DataFrame
            df = pd.DataFrame(table)

            # If the first row contains column names, use it as the header
            if len(df) > 0:
                df.columns = df.iloc[0]
                df = df.iloc[1:]  # remove the duplicate header row

            print(df)
            print("-" * 50)

# Close the PDF
pdf.close()

# Record the end time and compute the total duration
end_time = time.time()
print(f"\nTotal time to extract PDF tables: {end_time - start_time:.2f} seconds")
