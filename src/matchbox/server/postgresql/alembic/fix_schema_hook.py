import os
import re
import sys

path = sys.argv[1]
schema_name = os.environ.get("MB__SERVER__POSTGRES__DB_SCHEMA", "mb")
content = open(path).read()
content = content.replace(f'schema="{schema_name}"', "schema=schema")
content = re.sub(rf'"{re.escape(schema_name)}\.([^"]+)"', r'f"{schema}.\1"', content)
open(path, "w").write(content)
