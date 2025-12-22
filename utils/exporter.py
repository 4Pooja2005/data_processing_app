def export_data(df, file_path, file_type):
    if file_type == "csv":
        df.to_csv(file_path, index=False)
    elif file_type == "excel":
        df.to_excel(file_path, index=False)
    elif file_type == "json":
        df.to_json(file_path, orient="records", indent=2)
