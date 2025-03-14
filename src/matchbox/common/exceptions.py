"""Custom exceptions for Matchbox."""

from typing import Any

# -- Configuration exceptions


class MatchboxClientSettingsException(Exception):
    """Incorrect configuration provided to client."""


# -- Client-side API exceptions --


class MatchboxUnparsedClientRequest(Exception):
    """The API could not parse the content of the client request."""

    def __init__(
        self,
        message: str | None = None,
    ):
        """Initialise the exception."""
        if message is None:
            message = "The API could not parse the content of the client request"

        super().__init__(message)


class MatchboxUnhandledServerResponse(Exception):
    """The API sent an unexpected response."""

    def __init__(
        self,
        message: str | None = None,
    ):
        """Initialise the exception."""
        if message is None:
            message = "The API sent an unexpected response"

        super().__init__(message)


# -- Source exceptions --


class MatchboxSourceColumnError(Exception):
    """Source columns diverge with the warehouse."""


class MatchboxSourceEngineError(Exception):
    """Engine must be available in Source."""


class MatchboxSourceTableError(Exception):
    """Tables not found in your source data warehouse."""

    def __init__(
        self,
        message: str | None = None,
        table_name: str | None = None,
    ):
        """Initialise the exception."""
        if message is None:
            message = "Table doesn't exist in your source data warehouse."
            if table_name is not None:
                message += f"\nTable name: {table_name}"

        super().__init__(message)
        self.table_name = table_name


class MatchboxServerFileError(Exception):
    """There was a problem with file upload."""

    def __init__(self, message: str | None = None):
        """Initialise the exception."""
        if message is None:
            message = "There was a problem with file upload."

        super().__init__(message)


# -- Resource not found on server exceptions --


class MatchboxResolutionNotFoundError(Exception):
    """Resolution not found."""

    def __init__(self, message: str | None = None, resolution_name: str | None = None):
        """Initialise the exception."""
        if message is None:
            message = "Resolution not found."
            if resolution_name is not None:
                message = f"Resolution {resolution_name} not found."

        super().__init__(message)
        self.resolution_name = resolution_name


class MatchboxSourceNotFoundError(Exception):
    """Source not found on the server."""

    def __init__(
        self,
        message: str = None,
        address: str | None = None,
    ):
        """Initialise the exception."""
        if message is None:
            message = "Source not found on matchbox."
            if address:
                message = f"Source ({address}) not found."

        super().__init__(message)
        self.address = address


class MatchboxDataNotFound(Exception):
    """Data doesn't exist in the Matchbox source table."""

    def __init__(
        self,
        message: str | None = None,
        table: str | None = None,
        data: Any | None = None,
    ):
        """Initialise the exception."""
        if message is None:
            message = "Data doesn't exist in Matchbox."
            if table is not None:
                message += f"\nTable: {table}"
            if data is not None:
                message += f"\nData: {str(data)}"

        super().__init__(message)
        self.table = table
        self.data = data


# -- Server-side API exceptions --


class MatchboxClientFileError(Exception):
    """There was a problem with file download."""

    def __init__(self, message: str | None = None):
        """Initialise the exception."""
        if message is None:
            message = "There was a problem with file download."

        super().__init__(message)


class MatchboxConnectionError(Exception):
    """Connection to Matchbox's backend database failed."""


class MatchboxDeletionNotConfirmed(Exception):
    """Deletion must be confirmed: if certain, rerun with certain=True."""

    def __init__(self, message: str | None = None, children: list[str] | None = None):
        """Initialise the exception."""
        if message is None:
            message = "Deletion must be confirmed: if certain, rerun with certain=True."

        if children is not None:
            children_names = ", ".join(children)
            message = (
                f"This operation will delete the resolutions {children_names}, "
                "as well as all probabilities they have created. \n\n"
                "It won't delete validation associated with these "
                "probabilities. \n\n"
                "If you're sure you want to continue, rerun with certain=True. "
            )

        super().__init__(message)
