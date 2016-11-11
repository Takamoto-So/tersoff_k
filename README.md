pair tersoff_k
==============

_So Takamoto_

Interatomic potential for Si oxidation.
For potential, please see <http://dx.doi.org/10.1063/1.4965863>

Installation
------------

The `pair_style tersoff/k`, the `fix qeq/tersoff/k` are included
in the LAMMPS distribution as the USER-TERSOFF-K package.

To compile:
  cp `USER-TERSOFF-K` `lammps/src`
  cd `lammps/src`
  make yes-USER-TERSOFF-K
  make 'machine'


Documentation
-------------

The usage of `pair_style` and `pair_coeff` is same to tersoff potential.

The usage of `fix qeq/tersoff/k` is almose same to fix qeq/reax.
However, the 2nd and 3rd arguments (cutlo and cuthi) are not used.

Other
-----

We have tested this package in LAMMPS 16-Feb-16 version.

It may not work in some environments/versions.

Please understand that we cannot answer questions.