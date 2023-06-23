from src.data import utils as du

# import logging

# from sklearn.model_selection import train_test_split
# from dotenv import find_dotenv, load_dotenv
# import click


def data_hub_read(sample: int = None):
    """
    Read in Data Hub data
    Args:
        sample [int, default: None]: A size of random sample to draw
    Returns: dataset of dit.data_hub__companies
    """

    limit = ""

    if sample is not None:
        limit = f"order by random() limit {sample}"

    dsname = "dit.data_hub__companies"

    cols = """
        id::text as unique_id,
        company_number,
        name as company_name,
        string_to_array(btrim(trading_names, '[]'), ', ') as secondary_names,
        address_1,
        address_2,
        address_town,
        address_county,
        address_country,
        address_postcode as postcode,
        registered_address_1,
        registered_address_2,
        registered_address_town,
        registered_address_county,
        registered_address_country,
        registered_address_postcode as postcode_alt,
        uk_region,
        sector,
        description
    """

    query = f"""
        select {cols}
        from {dsname}
        where archived is False
        {limit}
    """

    df_dh = du.query(sql=query)

    return df_dh


def comp_house_read(sample: int = None):
    """
    Read in Companies House companies data
    Args:
        sample [int, default: None]: A size of random sample to draw
    Returns: dataset of companieshouse.companies
    """

    limit = ""

    if sample is not None:
        limit = f"order by random() limit {sample}"

    dsname = "companieshouse.companies"

    cols = """
        id::text as unique_id,
        company_number,
        company_name,
        array_remove(
            array[
                previous_name_1,
                previous_name_2,
                previous_name_3,
                previous_name_4,
                previous_name_5,
                previous_name_6
            ],
            ''
        ) as secondary_names,
        company_status,
        account_category,
        address_line_1,
        address_line_2,
        post_town,
        county,
        country,
        postcode,
        sic_code_1,
        sic_code_2,
        sic_code_3,
        sic_code_4
    """

    query = f"""
        select {cols}
        from {dsname}
        {limit}
    """

    df_ch = du.query(sql=query)

    return df_ch


def hmrc_exporters_read(sample: int = None):
    """
    Read in HMRC exporters company data
    Args:
        sample [int, default: None]: A size of random sample to draw
    Returns: dataset of hmrc.trade__exporters
    """

    limit = ""

    if sample is not None:
        limit = f"order by random() limit {sample}"

    dsname = "hmrc.trade__exporters"

    cols = """
        id::text as unique_id,
        null as company_number,
        company_name,
        null as secondary_names,
        postcode
    """

    query = f"""
        select {cols}
        from {dsname}
        {limit}
    """

    df_ex = du.query(sql=query)

    return df_ex


def export_wins_read(sample: int = None):
    """
    Read in Export Wins companies data
    Args:
        sample [int, default: None]: A size of random sample to draw
    Returns: dataset of dit.export_wins__wins_dataset
    """

    limit = ""

    if sample is not None:
        limit = f"order by random() limit {sample}"

    dsname = "dit.export_wins__wins_dataset"

    cols = """
        id::text as unique_id,
        cdms_reference as company_number,
        company_name,
        null as secondary_names,
        null as postcode
    """

    query = f"""
        select {cols}
        from {dsname}
        {limit}
    """

    df_ew = du.query(sql=query)

    return df_ew


# @click.command()
# @click.option(
#     "--output_name", required=True, type=str, help="Name of dataset to be saved"
# )
# @click.option(
#     "--split_test_set",
#     is_flag=True,
#     help="Whether to create separate test set",
# )
# def main(output_name, split_test_set):
#     """
#     Entrypoint
#     """
#     download_data_workspace_extract(output_name, split_test_set)


# if __name__ == "__main__":
#     log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     load_dotenv(find_dotenv())

#     main()
