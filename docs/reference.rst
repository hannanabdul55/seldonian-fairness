Reference
=========

===================
Seldonian Algorithm
===================

.. automodule:: seldonian.algorithm
      :members:
      :show-inheritance:
      :private-members: _safetyTest

==========================
Seldonian Abstract classes
==========================
Use this as a base class to implement your own fair model using the Seldonian approach.

.. automodule:: seldonian.seldonian
      :members:
      :show-inheritance:
      :special-members:
      :private-members: _safetyTest

===========================
Sample constraint functions
===========================

.. automodule:: seldonian.objectives
      :members:
      :show-inheritance:

===============================
CMA-ES optimizer implementation
===============================
.. automodule:: seldonian.cmaes
      :members:
      :show-inheritance:

=================================
Confidence bounds and validation
=================================
Non-asymptotic bounds on the mean of a bounded random variable used by the safety test,
and the exact-enumeration tools used to validate them (see ``reports/bounds/README.md``).

.. automodule:: seldonian.bounds
      :members:
      :show-inheritance:

.. automodule:: seldonian.bounds_eval
      :members:
