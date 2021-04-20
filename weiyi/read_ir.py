import re
import numpy as np
from numpy.linalg import norm
from fractions import Fraction
from monty.serialization import loadfn


def str2coo(coo_str: str):
    """
    Input a coordinate string and return an array. The string must have form: 0,0,0 or 1/3,1/3,0, etc.
    Args:
        coo_str (str): string of coordinates.
    """
    return [Fraction(s).__float__() for s in coo_str.split(',')]


class Outcar:
    """
    A class that extracts space group information from OUTCAR.
    """
    def __init__(self, filename='OUTCAR', uniform=False):
        header_pattern = r"irot\s+det\(A\)\s+alpha\s+n_x\s+n_y\s+n_z\s+tau_x\s+tau_y\s+tau_z"
        row_pattern = r"\s+\d+((?:\s+[\d\.\-]+)+)"
        # footer_pattern = r"-{104}"
        if not uniform:
            footer_pattern = r"-{77,}"
        else:
            footer_pattern = r"\s+Subroutine IBZKPT returns following result:"

        with open(filename, 'rt') as f:
            text = f.read()

        table_pattern_text = header_pattern + r"\s*^(?P<table_body>(?:\s+" + row_pattern + r")+)\s+" + footer_pattern
        table_pattern = re.compile(table_pattern_text, re.MULTILINE | re.DOTALL)
        operators = []
        for mt in table_pattern.finditer(text):
            table_body_text = mt.group("table_body")
            for line in table_body_text.split("\n"):
                entries = re.findall(r"[\d\-.]+", line)
                op = [float(entry) for entry in entries[1:]]
                operators.append(op)

        self._operators = np.array(operators)
        self._generate_mat()

    def _generate_mat(self):
        operations = []
        for i, op in enumerate(self._operators):
            det, theta, nx, ny, nz, tx, ty, tz = op
            assert np.isclose(nx ** 2 + ny ** 2 + nz ** 2, 1)
            theta = np.pi * theta / 180
            nbar_L = np.array([[0., -nz, ny],
                               [nz, 0., -nx],
                               [-ny, nx, 0.]])
            nbar_S = np.array([[nz, nx - 1j * ny],
                               [nx + 1j * ny, -nz]])
            # Rodrigues' rotation formula
            RL = det * (np.eye(3) + np.sin(theta) * nbar_L + (1 - np.cos(theta)) * (nbar_L @ nbar_L))
            RS = np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * nbar_S
            assert np.isclose(abs(np.linalg.det(RL)), 1, atol=1e-3)
            t = np.array([tx, ty, tz])
            operations.append([RL, RS, t])

        self.operation_matrix = operations


class Kpoint:
    def __init__(self, filename='KPOINTS'):
        with open(filename, 'rb') as f:
            text = f.read()

        if b'\nline' in text or b'\nLine' in text:
            row = re.search(b"\n *\d+", text).group().decode("utf-8")
            divisions = re.findall(r"\d+", row)[0]
            k_step = int(divisions)

            k_coo, k_sym = [], []
            for kp in re.findall(b" ?[\d. ]+[\d. ]+[\d. ]+ ?[! ]+.+", text):

                kp_str = [s.group() for s in re.finditer(b"[\d.a-zA-Z]+", kp)]
                coo = [float(s) for s in kp_str[:3]]
                k_coo.append(coo)
                sym = kp_str[-1].decode("utf-8")
                if 'G' in sym:
                    k_sym.append('\u0393')
                else:
                    k_sym.append(sym)

            self.hsk, idx0 = np.unique(k_coo, return_index=True, axis=0)
            self.hsk_sym = np.array(k_sym)[idx0]
            krange = np.arange(len(k_sym) / 2 * k_step)
            idx_from_all_kp = krange[(krange % k_step == k_step - 1) | (krange % k_step == 0)]
            self._idx_from_all_kp = idx_from_all_kp[idx0].astype(int)

        else:
            idx, k_coo, k_sym = [], [], []
            for i, row in enumerate(text.split(b'\n')[3:]):
                if not row:
                    break

                cont = [s for s in row.split(b' ') if s]
                if len(cont) == 5:
                    k_coo.append([float(s) for s in cont[:3]])
                    sym = cont[-1].decode("utf-8")
                    k_sym.append('\u0393' if 'G' in sym else sym)
                    idx.append(i)

            self.hsk, idx0 = np.unique(k_coo, return_index=True, axis=0)
            self.hsk_sym = np.array(k_sym)[idx0]
            self._idx_from_all_kp = np.array(idx)[idx0]


class Wavecar:
    """
    This class reads related information from WAVECAR. Adapted from pymatgen.io.vasp.outputs.Wavecar.
    Right now only supports spinless results.
    """
    def __init__(self, filename='WAVECAR'):
        # physical constant 2m/hbar**2
        self._C = 0.262465831
        with open(filename, 'r') as f:
            # read the header information
            recl, spin, rtag = np.fromfile(f, dtype=np.float64, count=3).astype(int)
            recl8 = int(recl / 8)
            self.spin = spin
            # assert rtag in (45200, 45210, 53300, 53310)
            assert rtag in (45200, 53310)

            # padding to end of fortran REC=1
            np.fromfile(f, dtype=np.float64, count=(recl8 - 3))

            # extract kpoint, bands, energy, and lattice information
            self.nkpoint, self.nband, self.encut = np.fromfile(f, dtype=np.float64, count=3).astype(np.int)
            a = np.fromfile(f, dtype=np.float64, count=9).reshape((3, 3))
            self.efermi = np.fromfile(f, dtype=np.float64, count=1)[0]

            self.vol = np.dot(a[0], np.cross(a[1], a[2]))

            # calculate reciprocal lattice
            b = 2 * np.pi * np.array([np.cross(a[1], a[2]),
                                      np.cross(a[2], a[0]),
                                      np.cross(a[0], a[1])]) / self.vol
            self.a = a
            self.b = b
            self._metric = b @ b.T
            # 2m/hbar^2 in agreement with VASP
            self._ksqrtcut = self.encut * self._C
            self._get_gmax()
            ranges = [np.roll(np.arange(-imax, imax + 1), -imax) for imax in self._gmax]
            self._grid = np.array(np.meshgrid(*ranges, indexing='ij')).T.reshape(-1, 3)
            # padding to end of fortran REC=2
            np.fromfile(f, dtype=np.float64, count=recl8 - 13)

            # reading records
            kpoints, gvecs, coeffs, band_energy = [], [], [], []
            for spin_idx in range(spin):
                spin_data = []
                spin_energy = []
                for k_idx in range(self.nkpoint):
                    k_data = []
                    nplane = int(np.fromfile(f, dtype=np.float64, count=1)[0])
                    kpoint = np.fromfile(f, dtype=np.float64, count=3)

                    if spin_idx == 0:
                        kpoints.append(kpoint)
                        gvec = self._get_gvecs(kpoint, nplane)
                        gvecs.append(gvec)

                    enocc = np.fromfile(f, dtype=np.float64, count=3 * self.nband).reshape((self.nband, 3))
                    spin_energy.append(enocc)

                    # padding to end of record that contains nplane, kpoints, evals and occs
                    np.fromfile(f, dtype=np.float64, count=(recl8 - 4 - 3 * self.nband) % recl8)

                    # extract coefficients
                    for band in range(self.nband):
                        data = np.fromfile(f, dtype=np.complex64, count=nplane)
                        np.fromfile(f, dtype=np.float64, count=recl8 - nplane)

                        assert len(data) == nplane
                        k_data.append(data)

                    spin_data.append(np.array(k_data))

                coeffs.append(spin_data)
                band_energy.append(np.array(spin_energy))

            self.kpoints = np.array(kpoints)
            self.gvecs = gvecs
            if self.spin == 1:
                self.coeffs = coeffs
            else:
                self.coeffs = [np.stack((up, down), axis=-1) for up, down in zip(*coeffs)]
            self.band_energy = band_energy

    def _get_gmax(self):
        """
        Find maximum absolute value of g vector along three axes. Algorithm adapted from WaveTrans.
        """
        b = self.b
        blen = np.linalg.norm(b, axis=1)
        recip_vol = (2 * np.pi) ** 3 / self.vol

        theta2 = np.arccos(np.dot(b[0], b[1]) / (blen[0] * blen[1]))
        s2 = recip_vol / (blen[2] * np.linalg.norm(np.cross(b[0], b[1])))
        theta1 = np.arccos(np.dot(b[0], b[2]) / (blen[0] * blen[2]))
        s1 = recip_vol / (blen[1] * np.linalg.norm(np.cross(b[0], b[2])))
        theta0 = np.arccos(np.dot(b[1], b[2]) / (blen[1] * blen[2]))
        s0 = recip_vol / (blen[0] * np.linalg.norm(np.cross(b[1], b[2])))
        det = np.min([[np.sin(theta0), s0, s0], [s1, np.sin(theta1), s1], [s2, s2, np.sin(theta2)]], axis=0)

        self._gmax = (np.sqrt(self._ksqrtcut) / blen / det).astype(int) + 1

    def _get_gvecs(self, kpoint, nplane):
        """
        Helper function find all g vectors of a kpoint. Note that the number of g vectors
        that has energy strictly less than or equal to ENCUT is usually less than the number
        of plane waves used in VASP. So a loop readjusting ENCUT is used to generate correct number of g vectors.
        Args:
            kpoint (np.ndarray): kpoint coordinates
            nplane (int): number of plane waves actually being used

        Returns:
            an array of all valid g vectors
        """
        ksqrts = ((self._grid + kpoint).dot(self._metric) * (self._grid + kpoint)).sum(-1)
        ksqrts_sorted = np.sort(ksqrts)
        is_g_included = ksqrts <= ksqrts_sorted[nplane - 1]
        return self._grid[is_g_included]

    def as_dict(self):
        d = {
            'lattice_vectors': self.a,
            'reciprocal_lattice_vectors': self.b,
            'coeffs': self.coeffs,
            'efermi': self.efermi,
            'band_energy': self.band_energy,
            'encut': self.encut,
            'gvecs': self.gvecs,
            'kpoints': self.kpoints,
            'nband': self.nband,
            'nkpoint': self.nkpoint,
            'spin': self.spin
        }

        return d


class BandCharacter(Wavecar):
    """
    This class calculate band characters using coefficients read from WAVECAR.

    Args:
        output_dir (str): directory where WAVECAR is stored.
        sg_number (int): space group number of the structure.
        gamma_only (bool): if the calculation is for gamma only.
    """
    def __init__(self, output_dir: str, sg_number, gamma_only=False):
        super(BandCharacter, self).__init__(filename=output_dir + '/WAVECAR')
        from pymatgen.core.structure import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        struc = Structure.from_file(output_dir + '/POSCAR')
        sa = SpacegroupAnalyzer(struc, symprec=0.1)
        struc_conv = sa.get_conventional_standard_structure()
        a0 = struc.lattice.matrix
        a1 = struc_conv.lattice.matrix
        mat = sa.get_conventional_to_primitive_transformation_matrix()
        invmat = np.linalg.inv(mat)
        self.mat = mat
        norm0 = norm(invmat @ a0, axis=1)
        norm1 = norm(a1, axis=1)
        ratio = norm0 / norm1
        if np.allclose(norm0, norm1) or np.allclose(ratio, ratio.round()):
            swap = None
        elif np.allclose(norm0[[1, 0]], norm1[[0, 1]]):
            swap = 'xy'
        elif np.allclose(norm0[[2, 0]], norm1[[0, 2]]):
            swap = 'xz'
        elif np.allclose(norm0[[2, 1]], norm1[[1, 2]]):
            swap = 'yz'
        else:
            raise ValueError
        self.swap = swap

        self.sg_number = sg_number
        self.little_group_dict = loadfn('ir_data/{}.json'.format(self.sg_number))

        if not gamma_only:
            hsk, hsk_sym, _idx_from_all_kp = [], [], []
            for key in self.little_group_dict.keys():
                sym, coo = key.split('&')
                coo = np.array(str2coo(coo))
                _bool = ((self.kpoints - coo) ** 2 < 1e-3).all(-1)
                if _bool.any():
                    hsk.append(coo)
                    hsk_sym.append(sym)
                    _idx_from_all_kp.append(_bool.nonzero()[0][0])
                else:
                    continue
            self.hsk = np.array(hsk)
            self.hsk_sym = hsk_sym
            self._idx_from_all_kp = np.array(_idx_from_all_kp)
        else:
            self.hsk = np.array([[0, 0, 0]])
            self.hsk_sym = ['GM']
            _bool = ((self.kpoints - self.hsk) ** 2 < 1e-3).all(-1)
            self._idx_from_all_kp = _bool.nonzero()[0][:1]

    @staticmethod
    def swap_ops_axis(ops, swap=None):
        """
        Swap acting axes of symmetry operations. Useful when the lattice vectors alignment is not the standard one.
        Args:
            ops (numpy.ndarray): symmetry operations.
            swap (str): which two axes to swap. One of 'xy', 'yz' and 'xz'.

        Returns: Swapped operations.

        """
        if swap is not None:
            if swap == 'xy':
                return ops[:, [1, 0, 2], :][..., [1, 0, 2, 3]]
            elif swap == 'xz':
                return ops[:, [2, 1, 0], :][..., [2, 1, 0, 3]]
            else:
                return ops[:, [0, 2, 1], :][..., [0, 2, 1, 3]]
        else:
            return ops

    def get_band_character(self, encut=100, en_tol=0.002):
        ksqrtcut = encut * self._C
        all_hsk_band_info = {}
        for hsk_sym, hsk, idx in zip(self.hsk_sym, self.hsk, self._idx_from_all_kp):
            gvec = self.gvecs[idx]
            coeff = self.coeffs[0][idx]
            assert gvec.shape[0] == coeff.shape[-1]
            band_en_occ = self.band_energy[0][idx]

            band_energy = band_en_occ[:, 0]
            is_border = (band_energy[1:] - band_energy[:-1]) < en_tol
            band_energy_border = np.concatenate(([0], np.arange(1, len(band_energy))[~is_border]))

            is_g_included_encut = ((gvec + hsk).dot(self._metric) * (gvec + hsk)).sum(-1) <= ksqrtcut
            gvec = gvec[is_g_included_encut]
            coeff = coeff[:, is_g_included_encut]
            coeff = coeff / np.linalg.norm(coeff, axis=-1, keepdims=True)

            for k, d in self.little_group_dict.items():
                if k.split('&')[0] == hsk_sym:
                    ops = d['matrix representations']
                    ops_swapped = self.swap_ops_axis(ops, self.swap)
                    band_char = []
                    for op in ops_swapped:
                        r, v = op[..., :3], op[..., 3]
                        gvec_rot = (self.mat @ (
                                    (np.linalg.inv(self.mat) @ (gvec + hsk).T).T @ np.linalg.inv(r)).T).T - hsk
                        compare = ((gvec_rot[:, np.newaxis, :] - gvec) ** 2 < 1e-3).all(-1)
                        if not compare.any(-1).all():
                            raise ValueError
                        rot_idx = compare.nonzero()[1]
                        coeff_rot = coeff[:, rot_idx]
                        phase = np.exp(-1j * 2 * np.pi * ((gvec_rot + hsk) @ v))
                        character = (np.conj(coeff_rot) * coeff * phase).sum(-1)
                        band_char.append(character)

                    band_char = np.column_stack(band_char)
                    band_char = np.add.reduceat(band_char, band_energy_border).round(decimals=3)
                    band_en_occ_reduced = band_en_occ[band_energy_border]

                    char_table = d['characters_real'] + 1j * d['characters_imag']
                    splt = (band_char @ char_table / len(char_table)).real
                    irs = []
                    for sp in splt:
                        sp = sp.round(decimals=1)
                        if np.allclose((b := sp.astype(int)), sp):
                            ir = ','.join([str(n) + symb for n, symb in zip(b, d['irrep symbols']) if n])
                            irs.append(ir)
                        else:
                            irs.append('?')

                    all_hsk_band_info[hsk_sym] = {
                        'n_irreps': len(d['irrep symbols']),
                        'n_levels': len(band_energy_border),
                        'k_coordinates': hsk,
                        'band_char': band_char,
                        'band_energy': band_en_occ_reduced[:, 0],
                        'band_occupation': band_en_occ_reduced[:, 2].round(decimals=3).astype(int),
                        'band_irrep': irs,
                        'band_degeneracy': band_char[:, 0].real.astype(int)}
            all_hsk_band_info['efermi'] = self.efermi
        return all_hsk_band_info
