"""Custom exceptions for Matchbox."""

from typing import Any

from pyarrow import Schema

from matchbox.common.graph import ResolutionName

# -- Common data objects exceptions --


class MatchboxArrowSchemaMismatch(Exception):
    """Arrow schema mismatch."""

    def __init__(self, expected: Schema, actual: Schema):
        """Initialise the exception."""
        message = f"Schema mismatch. Expected:\n{expected}\nGot:\n{actual}"

        super().__init__(message)


# -- Configuration exceptions --


class MatchboxClientSettingsException(Exception):
    """Incorrect configuration provided to client."""

    def __init__(
        self,
        message: str | None = None,
    ):
        """Initialise the exception."""
        if message is None:
            message = "Incorrect configuration provided to client."

        super().__init__(message)


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

    def __init__(self, http_status: int, details: str | None = None):
        """Initialise the exception."""
        message = f"The API sent an unexpected response with status {http_status}"
        if details:
            message += f" and the following message: {details}"

        super().__init__(message)


class MatchboxEmptyServerResponse(Exception):
    """The server returned an empty response when data was expected."""

    def __init__(self, message: str | None = None, operation: str | None = None):
        """Initialise the exception."""
        if message is None:
            message = "The server returned an empty response when data was expected."
            if operation is not None:
                message = (
                    f"The {operation} operation returned no data from the server "
                    "when data was expected."
                )

        super().__init__(message)
        self.operation = operation


# -- SourceConfig exceptions --


class MatchboxSourceFieldError(Exception):
    """Specified fields diverge with the warehouse."""


class MatchboxSourceClientError(Exception):
    """Location client must be set."""


class MatchboxSourceExtractTransformError(Exception):
    """Invalid ETL logic detected."""

    def __init__(
        self,
        logic: str | None = None,
    ):
        """Initialise the exception."""
        message = "Invalid ETL logic detected."
        if logic is not None:
            message += f"\n{logic}"

        super().__init__(message)


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


# -- ModelConfig exceptions --


class MatchboxModelConfigError(Exception):
    """There was a problem with ModelConfig."""

    def __init__(self, message: str | None = None):
        """Initialise the exception."""
        if message is None:
            message = "There was a problem with ModelConfig."

        super().__init__(message)


# -- Resource not found on server exceptions --


class MatchboxUserNotFoundError(Exception):
    """User not found."""

    def __init__(self, message: str | None = None, user_id: str | None = None):
        """Initialise the exception."""
        if message is None:
            message = "User not found."
            if user_id is not None:
                message = f"User {user_id} not found."

        super().__init__(message)
        self.user_id = user_id


class MatchboxResolutionNotFoundError(Exception):
    """Resolution not found."""

    def __init__(self, message: str | None = None, name: ResolutionName | None = None):
        """Initialise the exception."""
        if message is None:
            message = "Resolution not found."
            if name is not None:
                message = f"Resolution {name} not found."

        super().__init__(message)
        self.name = name


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


class MatchboxResolutionAlreadyExists(Exception):
    """Resolution already exists."""


class MatchboxTooManySamplesRequested(Exception):
    """Too many samples have been requested from the server."""


# -- Adapter DB exceptions --


class MatchboxNoJudgements(Exception):
    """No judgements found in the database when required for operation."""


class MatchboxDatabaseWriteError(Exception):
    """Could not be written to the backend DB, likely due to a constraint violation."""
