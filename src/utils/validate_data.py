
import great_expectations as gx
import pandas as pd
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Validate the Telco Customer Churn dataset using
    Great Expectations 1.x.

    Checks:
    1. Required columns
    2. Business rules
    3. Numeric ranges
    4. Statistical properties
    5. Data consistency
    """

    print("🔍 Starting data validation with Great Expectations...")

    # Work on a copy so the original DataFrame is not modified
    validation_df = df.copy()

    # TotalCharges may contain blank strings in the raw dataset.
    # Convert it to numeric so Great Expectations can validate it.
    if "TotalCharges" in validation_df.columns:
        validation_df["TotalCharges"] = pd.to_numeric(
            validation_df["TotalCharges"],
            errors="coerce"
        )

    # Create an in-memory Great Expectations context
    context = gx.get_context(mode="ephemeral")

    # Connect Great Expectations to pandas
    data_source = context.data_sources.add_pandas(
        name="telco_pandas"
    )

    # Create a Data Asset
    data_asset = data_source.add_dataframe_asset(
        name="telco_dataframe"
    )

    # Create a Batch Definition
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "telco_batch"
    )

    # Create a Batch using our DataFrame
    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": validation_df}
    )

    print("   📋 Preparing validation expectations...")

    # Each item contains a readable name and an expectation
    expectations = [

        # SCHEMA VALIDATION
        (
            "customerID_exists",
            gx.expectations.ExpectColumnToExist(
                column="customerID"
            )
        ),
        (
            "customerID_not_null",
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column="customerID"
            )
        ),

        (
            "gender_exists",
            gx.expectations.ExpectColumnToExist(column="gender")
        ),
        (
            "Partner_exists",
            gx.expectations.ExpectColumnToExist(column="Partner")
        ),
        (
            "Dependents_exists",
            gx.expectations.ExpectColumnToExist(column="Dependents")
        ),
        (
            "PhoneService_exists",
            gx.expectations.ExpectColumnToExist(column="PhoneService")
        ),
        (
            "InternetService_exists",
            gx.expectations.ExpectColumnToExist(column="InternetService")
        ),
        (
            "Contract_exists",
            gx.expectations.ExpectColumnToExist(column="Contract")
        ),
        (
            "tenure_exists",
            gx.expectations.ExpectColumnToExist(column="tenure")
        ),
        (
            "MonthlyCharges_exists",
            gx.expectations.ExpectColumnToExist(column="MonthlyCharges")
        ),
        (
            "TotalCharges_exists",
            gx.expectations.ExpectColumnToExist(column="TotalCharges")
        ),

        # BUSINESS LOGIC VALIDATION
        (
            "gender_valid",
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="gender",
                value_set=["Male", "Female"]
            )
        ),
        (
            "Partner_valid",
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="Partner",
                value_set=["Yes", "No"]
            )
        ),
        (
            "Dependents_valid",
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="Dependents",
                value_set=["Yes", "No"]
            )
        ),
        (
            "PhoneService_valid",
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="PhoneService",
                value_set=["Yes", "No"]
            )
        ),
        (
            "Contract_valid",
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="Contract",
                value_set=[
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )
        ),
        (
            "InternetService_valid",
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="InternetService",
                value_set=["DSL", "Fiber optic", "No"]
            )
        ),

        # NUMERIC RANGE VALIDATION
        (
            "tenure_non_negative",
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="tenure",
                min_value=0
            )
        ),
        (
            "MonthlyCharges_non_negative",
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="MonthlyCharges",
                min_value=0
            )
        ),
        (
            "TotalCharges_non_negative",
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="TotalCharges",
                min_value=0
            )
        ),

        # STATISTICAL VALIDATION
        (
            "tenure_range",
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="tenure",
                min_value=0,
                max_value=120
            )
        ),
        (
            "MonthlyCharges_range",
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="MonthlyCharges",
                min_value=0,
                max_value=200
            )
        ),
        (
            "tenure_not_null",
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column="tenure"
            )
        ),
        (
            "MonthlyCharges_not_null",
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column="MonthlyCharges"
            )
        ),

        # DATA CONSISTENCY VALIDATION
        (
            "TotalCharges_vs_MonthlyCharges",
            gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
                column_A="TotalCharges",
                column_B="MonthlyCharges",
                or_equal=True,
                mostly=0.95
            )
        )
    ]

    print("   ⚙️ Running validation checks...")

    failed_expectations = []
    passed_checks = 0

    # Run each expectation against the Batch
    for name, expectation in expectations:
        result = batch.validate(expectation)

        if result.success:
            passed_checks += 1
        else:
            failed_expectations.append(name)

    # Print validation summary
    total_checks = len(expectations)
    failed_checks = len(failed_expectations)

    if failed_checks == 0:
        print(
            f"✅ Data validation PASSED: "
            f"{passed_checks}/{total_checks} checks successful"
        )
    else:
        print(
            f"❌ Data validation FAILED: "
            f"{failed_checks}/{total_checks} checks failed"
        )
        print("   Failed expectations:", failed_expectations)

    return failed_checks == 0, failed_expectations
