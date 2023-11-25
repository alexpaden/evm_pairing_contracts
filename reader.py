import polars as pl

# Path to the Parquet file
file_path = 'results/clean_output.parquet'

df_parquet = pl.read_parquet(file_path)
print(df_parquet)
print("a sample row:")
print(df_parquet['disassembled_code'][2])

