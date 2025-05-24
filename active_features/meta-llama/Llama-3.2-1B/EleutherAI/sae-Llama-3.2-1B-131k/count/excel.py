import pandas as pd
import json
from pathlib import Path

def extract_data_item(item):
    """Extract data from a single item in the JSON structure into a DataFrame format"""
    rows = []
    
    features = item.get("features", {})
    
    for layer, langs in features.items():
        for lang, feature_info in langs.items():
            for feature_id, details in feature_info.items():
                interpretation = details.get("interpretation", "")
                tokens = details.get("tokens", [])
                
                # Convert the tokens list to a string for better readability in Excel
                tokens_str = ", ".join(tokens)
                
                rows.append({
                    "Layer": layer,
                    "Lang": lang,
                    "Feature ID": int(feature_id),
                    "Interpretation": interpretation,
                    "Tokens": tokens_str,
                })
    
    return pd.DataFrame(rows)

def main():
    file_path = Path(__file__).parent
    result_path = file_path / "result.json"

    with open(result_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # Create Excel writer with xlsxwriter engine
    output_file = file_path / "result.xlsx"
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        
        # Process each item in the JSON data and create a separate sheet for each
        for i, item in enumerate(json_data):
            # Extract the sentence for more descriptive sheet name
            sentence = item.get("sentences", "")
            # Create a valid and concise sheet name
            sheet_name = f"Item_{i+1}"
            
            # If there's a sentence, use the first few words as part of the sheet name
            if sentence:
                # Extract first few words (maximum 20 characters)
                short_desc = sentence.split()[:3]
                short_desc = " ".join(short_desc)[:20]
                sheet_name = f"Item_{i+1}_{short_desc}"
                
            # Excel has a 31 character limit for sheet names
            sheet_name = sheet_name[:31]
            
            # Extract data for this item
            df = extract_data_item(item)
            
            # Write to Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Optional: Adjust column widths
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns):
                # Set the column width based on the maximum length in the column
                max_len = max(
                    df[col].astype(str).map(len).max(),  # max length of column data
                    len(col)  # length of column name
                ) + 2  # adding a little extra space
                
                # Excel has column width units that are different from characters
                # This is an approximate conversion
                worksheet.set_column(idx, idx, min(max_len, 100))  # cap at 100 for very long text
    
    print(f"Data successfully converted and saved to {output_file}")

if __name__ == "__main__":
    main()