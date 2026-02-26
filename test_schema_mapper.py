import pandas as pd
from schema_mapper.mapper import SchemaMapper

def run_tests():
    mapper = SchemaMapper(threshold=65.0)
    all_passed = True

    with open("log.txt", "w", encoding="utf-8") as f:
        f.write("--- Running Smart Schema Mapper Test Suite ---\n\n")

        # Test Case 1
        f.write("Test 1: Standard Naming Variations\n")
        source_1 = ["cust_name", "EmailAddress", "phone#", "DOB"]
        golden_1 = ["Customer Name", "Email", "Phone Number", "Date of Birth"]
        mapping_1 = mapper.match_columns(source_1, golden_1)
        f.write(f"Result: {mapping_1}\n\n")
        try:
            assert mapping_1.get("cust_name") == "Customer Name"
            assert mapping_1.get("EmailAddress") == "Email"
            assert mapping_1.get("phone#") == "Phone Number"
        except AssertionError:
            f.write("FAIL: Test 1\n")
            all_passed = False

        # Test Case 2
        f.write("Test 2: Exact matches and Complete Mismatches\n")
        source_2 = ["ID", "weird_col_xyz", "revenue_2024"]
        golden_2 = ["ID", "System Log", "Revenue"]
        mapping_2 = mapper.match_columns(source_2, golden_2)
        f.write(f"Result: {mapping_2}\n\n")
        try:
            assert mapping_2.get("ID") == "ID"
            assert mapping_2.get("revenue_2024") == "Revenue"
            assert "weird_col_xyz" not in mapping_2
        except AssertionError:
            f.write("FAIL: Test 2\n")
            all_passed = False

        # Test Case 3
        f.write("Test 3: Duplicates and Order Insensitivity\n")
        source_3 = ["First Name", "Last Name", "Nm_First"]
        golden_3 = ["Name", "Surname"]
        mapping_3 = mapper.match_columns(source_3, golden_3)
        f.write(f"Result: {mapping_3}\n\n")
        try:
            assert "Name" in mapping_3.values(), "One of the source columns should map to 'Name'"
            assert list(mapping_3.values()).count("Name") == 1, "Only one column should map to 'Name'"
        except AssertionError:
            f.write("FAIL: Test 3\n")
            all_passed = False

        # Test Case 4
        f.write("Test 4: DataFrame Application\n")
        df = pd.DataFrame({
            "usr_addr": ["123 St", "456 Blvd"],
            "postcode": ["10001", "90210"],
            "unrelated": [1, 2]
        })
        golden_4 = ["User Address", "Postal Code", "City"]
        
        new_df, mapping_4 = mapper.apply_schema(df, golden_4)
        f.write(f"Mapped Columns:   {list(new_df.columns)}\n")
        f.write(f"Mapping Used:     {mapping_4}\n\n")
        try:
            assert "User Address" in new_df.columns
            assert "Postal Code" in new_df.columns
            assert "unrelated" in new_df.columns
        except AssertionError:
            f.write("FAIL: Test 4\n")
            all_passed = False

        if all_passed:
            f.write("All tests passed successfully! ✅\n")
        else:
            f.write("Some tests failed.\n")

if __name__ == "__main__":
    run_tests()
