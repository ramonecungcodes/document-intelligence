"""Second attempts at documents something complained about.

Importing the package registers every repairer, so `repair.base.build` can find one by
name from the manifest without the caller knowing which module defines it -- the same
arrangement every other plugin slot here uses.
"""
from repair import base            # noqa: F401
from repair import rerun           # noqa: F401
from repair import reprompt        # noqa: F401

from repair.base import REPAIRERS, Complaint, Context, Repaired, build, complaints_for

__all__ = ["REPAIRERS", "Complaint", "Context", "Repaired", "build",
           "complaints_for"]
