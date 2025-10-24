# Evaluation

## Python helpers

::: matchbox.client.eval
    options:
        show_root_heading: true
        show_root_full_path: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        filters:
            - "!^[A-Z]$"
            - "!^_"

## CLI module

::: matchbox.client.cli.eval
    options:
        show_root_heading: true
        show_root_full_path: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        show_submodules: true
        filters:
            - "!^[A-Z]$"
            - "!^_"
            - "!.*ui$"
