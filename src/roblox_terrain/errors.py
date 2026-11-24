"""Failures a caller can distinguish and recover from."""

from __future__ import annotations


class TerrainError(Exception):
    """Base type for this plugin."""


class ValidationError(TerrainError):
    """A parameter is outside the range the pipeline or Studio can honor."""


class DataUnavailable(TerrainError):
    """Elevation or land cover could not be loaded, and the command did not allow a download."""


class ConfirmationRequired(TerrainError):
    """The place would change, and the caller has not confirmed it."""


class ApplyError(TerrainError):
    """Studio accepted some of the edit and then a later chunk failed.

    The place may already contain a partial terrain. Read it before retrying.
    ``clear_generated_terrain`` removes ``Workspace.RealisticTerrain`` and the
    voxel box recorded on that folder.
    """


class PipelineError(TerrainError):
    """The grids were loaded, but a stage produced a result that cannot be applied."""
