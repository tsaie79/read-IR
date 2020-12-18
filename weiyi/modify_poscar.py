import numpy as np
from pymatgen.core.structure import Structure, SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

SG_INV = np.concatenate((np.array([2, 147, 148, 175, 176]), np.arange(10, 16), np.arange(47, 75),
                         np.arange(83, 89), np.arange(123, 143), np.arange(162, 168),
                         np.arange(191, 195), np.arange(200, 207), np.arange(221, 231)))


def modify(poscar, symprec=0.1):
    struc = Structure.from_file(poscar)
    sg_op = SpacegroupAnalyzer(struc, symprec=symprec).get_space_group_operations()
    inversion = SymmOp.from_xyz_string('-x, -y, -z')

    if inversion in sg_op:
        return struc
    else:
        s = struc.copy()
        # translate inversion center to cell center
        inv_center_trans = np.array([0.5, 0.5, 0.5]) - s.frac_coords.mean(0)
        for site in range(len(struc)):
            s.translate_sites(site, inv_center_trans)

        assert inversion in SpacegroupAnalyzer(s).get_space_group_operations()
        return s
