::: matchbox.common.eval
    options:
        show_root_heading: true
        show_root_full_path: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        extra:
            show_root_docstring: true
        filters:
            - "!^[A-Z]$"  # Excludes single-letter uppercase variables (like T, P, R)
            - "!^_"       # Excludes private attributes
            - "!_logger$"  # Excludes logger variables
            - "!_path$"    # Excludes path variables
            - "!model_config" # Excludes Pydantic configuration