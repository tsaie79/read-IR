import numpy as np
from fractions import Fraction
from monty.serialization import loadfn


def str2coo(coo_str: str):
    """
    Input a coordinate string and return an array. The string must have form: 0,0,0 or 1/3,1/3,0, etc.
    Args:
        coo_str (str): string of coordinates.
    """
    return [Fraction(s).__float__() for s in coo_str.split(',')]


class Wavecar:
    """
    This class reads related information from WAVECAR. Adapted from pymatgen.io.vasp.outputs.Wavecar
    """

    def __init__(self, filename='WAVECAR'):
        # super().__init__()
        # physical constant 2m/hbar**2
        self._C = 0.262465831
        with open(filename, 'r') as f:
            # read the header information
            recl, spin, rtag = np.fromfile(f, dtype=np.float64, count=3).astype(np.int)
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

    def _get_gvecs(self, kpoint, nplane, inc=0.05):
        """
        Helper function find all g vectors of a kpoint. Note that the number of g vectors
        that has energy strictly less than or equal to ENCUT is usually less than the number
        of plane waves used in VASP. So a loop readjusting ENCUT is used to generate correct number of g vectors.
        Args:
            kpoint (np.ndarray): kpoint coordinates
            nplane (int): number of plane waves actually being used
            inc (float): energy searching increment

        Returns:
            an array of all valid g vectors
        """
        ksqrtcut = self._ksqrtcut
        i = 0
        while True:
            if i > 10:
                raise ValueError('Can not find correct number of g vectors within'
                                 ' {:.3f} eV larger than encut.'.format(i * inc / self._C))
            is_g_included = ((self._grid + kpoint).dot(self._metric) * (self._grid + kpoint)).sum(-1) <= ksqrtcut
            if is_g_included.sum() == nplane:
                break

            assert is_g_included.sum() < nplane, "Try to decrease the searching increment."
            ksqrtcut += inc
            i += 1
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
    def __init__(self, output_dir, sg_number, gamma_only=False):
        super(BandCharacter, self).__init__(filename=output_dir / 'WAVECAR')
        self.sg_number = sg_number
        self.little_group_dict = loadfn('ir_data/{}.json'.format(self.sg_number))

        if not gamma_only:
            hsk, hsk_sym, _idx_from_all_kp = [], [], []
            for key in self.little_group_dict.keys():
                sym, coo = key.split('&')
                coo = str2coo(coo)
                _bool = np.isclose(self.kpoints, coo).all(-1)
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
            _bool = np.isclose(self.kpoints, self.hsk).all(-1)
            self._idx_from_all_kp = _bool.nonzero()[0][:1]

    def get_band_character(self, encut=50, en_tol=0.002):
        ksqrtcut = encut * self._C
        all_hsk_band_info = {}
        for hsk_sym, hsk, idx in zip(self.hsk_sym, self.hsk, self._idx_from_all_kp):
            gvec = self.gvecs[idx]
            coeff = self.coeffs[0][idx]
            assert gvec.shape[0] == coeff.shape[-1]
            band_en_occ = self.band_energy[0][idx]

            band_energy = band_en_occ[:, 0]
            is_border = np.abs(band_energy[1:] - band_energy[:-1] < en_tol)
            band_energy_border = np.concatenate(([0], np.arange(1, len(band_energy))[~is_border]))

            is_g_included_encut = ((gvec + hsk).dot(self._metric) * (gvec + hsk)).sum(-1) <= ksqrtcut
            gvec = gvec[is_g_included_encut]
            coeff = coeff[:, is_g_included_encut]
            coeff = coeff / np.linalg.norm(coeff, axis=-1, keepdims=True)

            for k, d in self.little_group_dict.items():
                if k.startswith(hsk_sym):
                    ops = d['matrix representations']

                    band_char = []
                    for op in ops:
                        r, v = op[..., :3], op[..., 3]
                        gvec_rot = (gvec + hsk) @ np.linalg.inv(r) - hsk
                        compare = np.isclose(gvec_rot[:, np.newaxis, :], gvec, atol=1e-3).all(-1)
                        if not compare.any(-1).all():
                            raise ValueError
                        rot_idx = compare.nonzero()[1]

                        coeff_rot = coeff[:, rot_idx]
                        phase = np.exp(-1j * 2 * np.pi * ((gvec_rot + hsk) @ v))
                        character = (np.conj(coeff_rot) * coeff * phase).sum(-1)
                        band_char.append(character)

                    band_char = np.column_stack(band_char)
                    band_char = np.add.reduceat(band_char, band_energy_border).round(decimals=3)
                    band_en_occ = band_en_occ[band_energy_border]

                    char_table = d['characters_real'] + 1j * d['characters_imag']
                    splt = (band_char @ char_table / len(char_table)).real
                    irs = []
                    for sp in splt:
                        sp = sp.round(decimals=2)
                        b = sp.astype(int)
                        if np.allclose(b, sp):
                            ir = ''.join([str(n) + symb if n else '' for n, symb in zip(b, d['irrep symbols'])])
                            irs.append(ir)
                        else:
                            irs.append('?')

                    all_hsk_band_info[hsk_sym] = {
                        'n_levels': len(band_energy_border),
                        'k_coordinates': hsk.tolist(),
                        'band_char': band_char.tolist(),
                        'band_energy': band_en_occ[:, 0].tolist(),
                        'band_occupation': band_en_occ[:, 2].round(decimals=3).astype(int).tolist(),
                        'band_irrep': irs
                    }

        return all_hsk_band_info
