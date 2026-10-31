"""NEST-Pakistan: a national energy model coupled to the NEST water/land
module of a river basin (the Indus by default), as a command-line workflow.

Modules, in pipeline order:

- :mod:`.config` -- ``config.yaml``: every assumption of a run.
- :mod:`.basin_data` -- input files a basin needs and their global sources.
- :mod:`.workflow` -- build the energy model, relax it, add the nexus, solve.
- :mod:`.calibration` -- pre-solve consistency check and documented fixes.
- :mod:`.checks` -- input/assumption checks with a PDF.
- :mod:`.report` -- result tables, CSVs, PDF reports, scenario comparison.
- :mod:`.cli` -- ``mix-models nest-pak ...``.

See README.md in this folder for the workflow and DATA_REQUIREMENTS.md for
setting up another river basin.
"""
