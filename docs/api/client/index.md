# API reference

`matchbox.client` is the client used to interact with the [Matchbox server](../../server/install.md).
The `match` and `query` functions in this file are also accessible from the top-level `matchbox` module.

::: matchbox.client
    options:
        show_root_heading: true
        show_root_full_path: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        filters:
            - "!^[A-Z]$"  # Excludes single-letter uppercase variables (like T, P, R)
            - "!^_"       # Excludes private attributes
            - "!_logger$"  # Excludes logger variables
            - "!_path$"    # Excludes path variables
            - "!app$"    # Excludes FastAPI app