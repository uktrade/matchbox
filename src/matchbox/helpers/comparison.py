import sqlglot.expressions as exp
from sqlglot import parse_one
from sqlglot.errors import ParseError


def comparison(sql_condition: str, dialect: str = "postgres") -> str:
    """
    Validates any number of SQL conditions and returns as valid dialect SQL
    appropriate for a WHERE clause.

    Requires all column references be explicitly declared as from "l" and
    "r" tables.
    """
    parsed_sql = parse_one(sql_condition)

    for node in parsed_sql.walk():
        if not isinstance(
            node[0], (exp.Connector, exp.Predicate, exp.Condition, exp.Identifier)
        ):
            raise ParseError(
                "Must be valid WHERE clause statements. " f"Found {type(node[0])}"
            )

    left = False
    right = False
    for column in parsed_sql.find_all(exp.Column):
        if column.table == "l":
            left = True
        elif column.table == "r":
            right = True
        elif column.table == "":
            raise ParseError(
                "Columns must be explicitly declared as one of table "
                '"l" or "r".\n\n'
                f"No table declared for column {column.name}."
            )
        else:
            raise ParseError(
                "Columns must be explicitly declared as one of table "
                '"l" or "r".\n\n'
                f"Found column {column.table}."
            )
    if not (left and right):
        raise ParseError('Conditions must reference both "l" and "r".')

    pg_sql = parsed_sql.sql(dialect=dialect)

    return pg_sql