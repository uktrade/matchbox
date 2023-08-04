from src.data import utils as du


class MakeDim(object):
    def __init__(self, unique_fields: list, fact_table: str, dim_table: str):
        self.unique_fields = ", ".join(unique_fields)
        self.fact_table = fact_table
        self.dim_table = dim_table

    def make_dim_table(self, overwrite: bool) -> None:
        if du.check_table_exists(self.dim_table) and not overwrite:
            raise ValueError(
                "Table exists. Set overwrite to True if you want to proceed."
            )

        if overwrite:
            sql = f"drop table if exists {self.dim_table};"
            du.query_nonreturn(sql)

        sql = f"""
            create table {self.dim_table} as (
                select
                    uuid_generate_v4() as dim_uuid,
                    {self.unique_fields}
                from (
                    select distinct
                        {self.unique_fields}
                    from
                        {self.fact_table}
                ) dedupe
            );
        """

        du.query_nonreturn(sql)
