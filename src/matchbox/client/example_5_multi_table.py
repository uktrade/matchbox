"""Join multiple datasets in a single source."""

from sqlalchemy import create_engine

from matchbox import pipelines as mbp

engine = create_engine("postgresql:///")
postgres = mbp.RelationalDBLocation(name="data_workspace_postgres", client=engine)
collection = mbp.Collection("default_companies")

# Unioning different tables
hmrc = mbp.Source(
    location=postgres,
    name="companies_house",
    extract_transform="""
        select distinct on (company_uktrade_id)
            company_uktrade_id id,
            company_name,
            postcode
        from
            hmrc.trade__exporters
        union
        select distinct on (company_uktrade_id)
            company_uktrade_id id,
            company_name,
            postcode
        from
            hmrc.trade__importers;
    """,
    key_field="id",
    index_fields=["company_name", "postcode"],
)


# Joining different tables
hmrc = mbp.Source(
    location=postgres,
    name="companies_house",
    extract_transform="""
        select 
            coalesce(u."hashedUuid", t."hashedUuid") as id,
            u."companyName" as company_name,
            u."created"::date as created,
            u."companyWebsite" as company_website,
            u."dunsNumber" as duns_number,
            u."addressLine1" as company_address_1,
            u."addressLine2" as company_address_2,
            u."town" as company_town,
            u."county" as company_county,
            u."postcode" as postcode,
            u."companyLocation" as country_iso2,
            t."location" as target_region,
            t."locationCity" as target_city,
            t."sectorID" as company_sector_id,
            u."fullName" as contact_name,
            u."role" as contact_role,
            u."email" as contact_email,
            u."telephoneNumber" as contact_number
        from "dbt"."great_gov_uk__expand_your_business_user" u
        full outer join "dbt"."great_gov_uk__expand_your_business_triage" t 
            on u."hashedUuid"::text = t."hashedUuid"::text;
    """,
    key_field="id",
    index_fields=[
        "company_name",
        "created",
        "company_website",
        "duns_number",
        "company_address_1",
        "company_address_2",
        "company_town",
        "company_county",
        "postcode",
        "country_iso2",
        "target_region",
        "target_city",
        "company_sector_id",
        "contact_name",
        "contact_role",
        "contact_email",
        "contact_number",
    ],
)
