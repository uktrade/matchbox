This page contains a list of known data and security caveats determined by the current implementation of Matchbox.

## Sensitive data sources

!!! warning
    Matchbox allows users with **read** permission on a collection to view all identifier values in that collection's indexed datasets.

### What's usually safe

Matchbox only stores:

* Cryptographically secure hashes of the original data
* Key fields, which is usually a standard identifier for your organisation (often not sensitive)

For example, a dataset of tax records may contain sensitive information, but since all businesses have tax records, the presence of a business in the dataset isn’t necessarily sensitive.

### What's risky

Problems arise when:

* A user accesses a source that maps identifiable data (like names and addresses) to secret identifiers from another source that shouldn't be revealed.
* Which identities appear in an indexed source is sensitive information in itself.

**Example:** Imagine a table of high-profile individuals under investigation. Matchbox doesn’t store investigation details, but it does expose the identifiers you use as key fields (e.g.: passport numbers). If another source links those keys to names, Matchbox could be used to reveal identities.

!!! tip
    Do not use Matchbox to index sources where the key fields are sensitive data.

## Table metadata

!!! warning
    Matchbox allows users with **read** permission to see the **list of fields** and extract-transform logic of an indexed source.

!!! tip
    Avoid indexing sources where the **field names** or **structure** are considered sensitive.

## Extract-transform logic

!!! danger
    **Malicious queries** could be embedded in source configs.

**Example**:

* A source from `RelationalDBLocation` stores SQL queries used during indexing.
* If an automated pipeline reuses these queries, a malicious user with **write** permission could inject harmful SQL (e.g., `DROP DATABASE`).

Matchbox performs **basic validation**, but **cannot guarantee query safety**.

!!! tip
    Ensure that clients running extract-transform logic stored on the server have the least priviliges required to run legitimate queries.

