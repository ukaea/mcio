MCIO
====

MCIO is a Python package for building Monte Carlo radiation transport models. It lets you:

- **Build CSG models in Python**: define surfaces, cells, materials, sources and tallies once, then write
  them to both MCNP and OpenMC input formats.
- **Convert CAD to CSG**: decompose STEP (AP242) CAD models into CSG cells, with materials assigned by
  colour or label, using multiple cores.
- **Convert CSG to CAD**: turn CSG volumes back into OCP solids and STEP files for inspection.
- **Debug models**: extract lost particles from MCNP output into a STEP file, to load into your CAD software.

.. warning::

   MCIO uses cm for length and MeV for energy, and requires STEP files to use the AP242 schema.

Start with :doc:`install`, then follow the :doc:`userguide`. The source code is on
`GitHub <https://github.com/ukaea/mcio>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   install
   userguide
   api
