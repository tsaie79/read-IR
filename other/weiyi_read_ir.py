import os
import re
import numpy as np


class Wavecar:
    """
    This class reads related information from WAVECAR. Adapted from pymatgen.io.vasp.outputs.Wavecar
    """

    def __init__(self, filename='WAVECAR'):
        # super().__init__()
        # physical constant 2m/hbar**2
        self._C = 0.262465831

        with open(filename, 'rb') as f:
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


class Outcar:
    """
    A class that extracts space group information from OUTCAR.
    """

    def __init__(self, filename='OUTCAR', uniform=False):
        # super().__init__()

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
        # super().__init__()
        with open(filename, 'rb') as f:
            text = f.read()

        if b'\nline' in text:
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


class IrBand(Wavecar, Outcar, Kpoint):
    def __init__(self, output_dir, gamma_only=False, uniform=False):
        os.chdir(output_dir)
        # super().__init__()
        Wavecar.__init__(self)
        Outcar.__init__(self, uniform=uniform)
        if gamma_only or uniform:
            self.hsk = np.array([[0, 0, 0]])
            self.hsk_sym = ['\u0393']
            _bool = np.isclose(self.kpoints, np.zeros(3)).all(-1)
            self._idx_from_all_kp = np.arange(len(self.kpoints))[_bool][:1]
        else:
            Kpoint.__init__(self)

        rot, trans, spin_rot = [], [], []

        at = self.a.T
        invat = np.linalg.inv(at)
        for op in self.operation_matrix:
            rot_op = invat @ op[0] @ at
            # rotation is an integer matrix in this basis
            assert np.allclose(rot_op, np.round(rot_op), atol=1e-3)
            trans_op = op[2]
            rot.append(rot_op)
            trans.append(trans_op)
            if self.spin == 2:
                spin_rot.append(op[1])

        self.rot = np.array(rot)
        self.trans = np.array(trans)
        if self.spin == 2:
            self.spin_rot = np.array(spin_rot)

        self._get_littlegroup()

        os.chdir('..')

    def get_band_character_v2(self, encut=50, output_list=False):
        ksqrtcut = encut * self._C
        all_hsk_band_info = {}

        if self.spin == 1:
            for hsk_sym, hsk, idx, lg in zip(self.hsk_sym, self.hsk, self._idx_from_all_kp, self.little_group):
                gvec = self.gvecs[idx]
                coeff = self.coeffs[0][idx]
                assert gvec.shape[0] == coeff.shape[-1]
                band_energy = self.band_energy[0][idx]
                band_energy_border = np.unique(band_energy[:, 0].round(decimals=3), return_index=True)[1]

                is_g_included_encut = ((gvec + hsk).dot(self._metric) * (gvec + hsk)).sum(-1) <= ksqrtcut
                # get gvec and coeff within searching encut
                gvec = gvec[is_g_included_encut]
                coeff = coeff[:, is_g_included_encut]
                # renormalize coeff
                coeff = coeff / np.linalg.norm(coeff, axis=-1, keepdims=True)

                # calculate R(k+G)
                # kp_rot = np.einsum('ijk,lk->ilj', rot, gvec + hsk)

                band_char = []

                for r, v in zip(*lg):
                    gvec_rot = (gvec + hsk) @ np.linalg.inv(r) - hsk
                    # comparison between rows of g and gvec
                    compare = np.isclose(gvec_rot[:, np.newaxis, :], gvec, atol=1e-3).all(-1)
                    # ensure each row of gvec_rot exists in gvec
                    assert compare.any(-1).all()
                    rot_idx = compare.nonzero()[1]

                    coeff_rot = coeff[:, rot_idx]
                    phase = np.exp(-1j * 2 * np.pi * ((gvec_rot + hsk) @ v))
                    character = (np.conj(coeff_rot) * coeff * phase).sum(-1)
                    band_char.append(character)

                band_char = np.column_stack(band_char)
                # sum up coefficients of bands with same energy
                band_char = np.add.reduceat(band_char, band_energy_border).real.round(decimals=3)
                is_char_int = np.allclose(band_char, band_char.round(decimals=2).astype(int), atol=1e-2)
                if is_char_int:
                    band_char = band_char.astype(int)

                band_energy = band_energy[band_energy_border]

                all_hsk_band_info[hsk_sym] = {
                    'n_levels': len(band_energy_border),
                    'k_coordinates': hsk.tolist() if output_list else hsk,
                    'little_group_operations': lg,
                    'is_char_int': is_char_int,
                    'band_char': band_char.tolist() if output_list else band_char,
                    'band_energy': band_energy[:, 0].tolist() if output_list else band_energy[:, 0],
                    'band_occupation': band_energy[:, 2].round(decimals=3).astype(int).tolist()
                    if output_list else band_energy[:, 2].round(decimals=3).astype(int)}

        elif self.spin == 2:
            for hsk_sym, hsk, idx, lg in zip(self.hsk_sym, self.hsk, self._idx_from_all_kp, self.little_group):
                gvec = self.gvecs[idx]
                coeff = self.coeffs[idx]
                assert gvec.shape[0] == coeff.shape[1]
                band_energy = self.band_energy[0][idx]
                band_energy_border = np.unique(band_energy[:, 0].round(decimals=3), return_index=True)[1]

                is_g_included_encut = ((gvec + hsk).dot(self._metric) * (gvec + hsk)).sum(-1) <= ksqrtcut
                # get gvec and coeff within searching encut
                gvec = gvec[is_g_included_encut]
                coeff = coeff[:, is_g_included_encut, :]

                coeff = coeff / np.linalg.norm(coeff, axis=1, keepdims=True)

                band_char = []

                for r, v, s in zip(*lg):
                    gvec_rot = (gvec + hsk) @ np.linalg.inv(r) - hsk
                    # comparison between rows of g and gvec
                    compare = np.isclose(gvec_rot[:, np.newaxis, :], gvec, atol=1e-3).all(-1)
                    # ensure each row of gvec_rot exists in gvec
                    assert compare.any(-1).all()
                    rot_idx = compare.nonzero()[1]

                    coeff_rot = coeff[:, rot_idx, :]
                    phase = np.exp(-1j * 2 * np.pi * ((gvec_rot + hsk) @ v))

                    character = (np.einsum('ijk,ijl,kl->ij', coeff_rot.conj(), coeff, s) * phase).sum(-1)
                    band_char.append(character)

                band_char = np.column_stack(band_char)
                band_char = np.add.reduceat(band_char, band_energy_border).real.round(decimals=3)
                is_char_int = np.allclose(band_char, band_char.round(decimals=2).astype(int), atol=1e-2)
                if is_char_int:
                    band_char = band_char.astype(int)

                band_energy = band_energy[band_energy_border]
                all_hsk_band_info[hsk_sym] = {
                    'n_levels': len(band_energy_border),
                    'k_coordinates': hsk.tolist() if output_list else hsk,
                    'little_group_operations': lg,
                    'is_char_int': is_char_int,
                    'band_char': band_char.tolist() if output_list else band_char,
                    'band_energy': band_energy[:, 0].tolist() if output_list else band_energy[:, 0],
                    'band_occupation': band_energy[:, 2].round(decimals=3).astype(int).tolist()
                    if output_list else band_energy[:, 2].round(decimals=3).astype(int)}

        return all_hsk_band_info

    def get_band_character(self, encut=50):
        ksqrtcut = encut * self._C
        all_hsk_band_info = {}
        for hsk_sym, hsk, idx, lg in zip(self.hsk_sym, self.hsk, self._idx_from_all_kp, self.little_group):
            spin_info = {'k_coordinates': hsk, 'little_group_operations': lg}

            for spin in range(self.spin):

                gvec = self.gvecs[idx]
                coeff = self.coeffs[spin][idx]
                assert gvec.shape[0] == coeff.shape[-1]
                band_energy = self.band_energy[spin][idx]
                band_energy_border = np.unique(band_energy[:, 0].round(decimals=3), return_index=True)[1]

                is_g_included_encut = ((gvec + hsk).dot(self._metric) * (gvec + hsk)).sum(-1) <= ksqrtcut
                # get gvec and coeff within searching encut
                gvec = gvec[is_g_included_encut]
                coeff = coeff[:, is_g_included_encut]
                # renormalize coeff
                coeff = coeff / np.linalg.norm(coeff, axis=-1, keepdims=True)

                # calculate R(k+G)
                # kp_rot = np.einsum('ijk,lk->ilj', rot, gvec + hsk)

                band_char = []
                for r, v in zip(*lg):
                    gvec_rot = (gvec + hsk) @ np.linalg.inv(r) - hsk
                    # comparison between rows of g and gvec
                    compare = np.isclose(gvec_rot[:, np.newaxis, :], gvec, atol=1e-3).all(-1)
                    # ensure each row of gvec_rot exists in gvec
                    assert compare.any(-1).all()
                    rot_idx = compare.nonzero()[1]

                    coeff_rot = coeff[:, rot_idx]
                    phase = np.exp(-1j * 2 * np.pi * ((gvec_rot + hsk) @ v))
                    character = (np.conj(coeff_rot) * coeff * phase).sum(-1)
                    band_char.append(character)

                band_char = np.column_stack(band_char)
                # sum up coefficients of bands with same energy
                band_char = np.add.reduceat(band_char, band_energy_border).real.round(decimals=3)
                is_char_int = np.allclose(band_char, band_char.round(decimals=2).astype(int), atol=1e-2)
                if is_char_int:
                    band_char = band_char.astype(int)

                band_energy = band_energy[band_energy_border]

                if self.spin == 1:

                    all_hsk_band_info[hsk_sym] = {'n_levels': len(band_energy_border),
                                                  'k_coordinates': hsk,
                                                  'little_group_operations': lg,
                                                  'is_char_int': is_char_int,
                                                  'band_char': band_char,
                                                  'band_energy': band_energy[:, 0],
                                                  'band_occupation': band_energy[:, 2].round(decimals=3).astype(int)}

                elif self.spin == 2:
                    spin_info['spin_up' if spin == 0 else 'spin_down'] = {'n_levels': len(band_energy_border),
                                                                          'is_char_int': is_char_int,
                                                                          'band_char': band_char,
                                                                          'band_energy': band_energy[:, 0],
                                                                          'band_occupation': band_energy[:, 2].round(
                                                                              decimals=3).astype(int)}

            all_hsk_band_info[hsk_sym] = spin_info

        return all_hsk_band_info

    def as_dict(self, output_list=False):
        d = {'lattice_vectors': self.a.tolist() if output_list else self.a,
             'reciprocal_lattice_vectors': self.b.tolist() if output_list else self.b,
             'efermi': self.efermi,
             'nband': self.nband,
             'nkpoint': self.nkpoint,
             'spin': self.spin,
             'band_symmetry': self.get_band_character(output_list)}

        return d

    def _get_littlegroup(self):
        little_group = []
        for hsk in self.hsk:
            # calculate Rk - k
            hsk_diff = hsk @ np.linalg.inv(self.rot) - hsk
            idx = np.isclose(hsk_diff, hsk_diff.round(decimals=3).astype(int), atol=1e-3).all(-1).nonzero()[0]
            if self.spin == 1:
                little_group.append([self.rot[idx], self.trans[idx]])
            else:
                little_group.append([self.rot[idx], self.trans[idx], self.spin_rot[idx]])

        self.little_group = little_group


if __name__ == '__main__':
    # wc = Wavecar('MoS2/WAVECAR')
    # outcar = Outcar('MoS2/OUTCAR', uniform=True)

    ir = IrBand('MoS2', uniform=True)
    info = ir.get_band_character_v2()
