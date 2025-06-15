# darcyflow

## Environment setup
* CI runs on both, Ubuntu and Windows. However, only the latest OS version runners are used.
* Requirement: python >= 3.10
* Steps:
   * Create and activate Python venv
   * Install the requirements

## Examples
* TPFA and MPFA solutions comparison. `$env:PYTHONPATH='.'; python .\examples\compare_methods.py`
    * In Linux: `PYTHONPATH='.' python .\examples\compare_methods.py`
    * Open compare_methods.py and try running with different permiability field from darcyflow/porus_media.py
    * Try setting sources and well in different places of the grid by updating pressure_bc. Positive value - source, negative value - well.
* Visualise examples of implemented permiability field generators: `$env:PYTHONPATH='.'; python .\examples\viz_permiability_fields.py`
