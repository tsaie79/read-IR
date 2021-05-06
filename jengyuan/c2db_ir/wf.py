import shutil

from atomate.vasp.workflows.base.core import get_wf
from atomate.vasp.database import VaspCalcDb
from fireworks import LaunchPad, Workflow
import os
from atomate.vasp.powerups import (
    add_additional_fields_to_taskdocs,
    preserve_fworker,
    add_modify_incar,
    set_queue_options,
    set_execution_options,
    clean_up_files,
    add_modify_kpoints
)
import numpy as np
from pymatgen.core.structure import Structure, SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.sets import MPRelaxSet
from subprocess import call
from pytopomat.workflows.fireworks import IrvspFW

from mpinterfaces.utils import ensure_vacuum

c2db = VaspCalcDb.from_db_file("/home/tug03990/scripts/read-IR/jengyuan/c2db_ir/c2db.json")
for e in list(c2db.collection.find({"magstate":"NM"}))[:1]:
    st = e["structure"]

    os.makedirs("symmetrized_st", exist_ok=True)
    os.chdir("symmetrized_st")
    st = Structure.from_dict(st)
    st = ensure_vacuum(st, 20)
    st.to("poscar", "POSCAR")
    call("phonopy --symmetry --tolerance 0.01 -c POSCAR".split(" "))
    st = Structure.from_file("PPOSCAR")
    st.to("poscar", "POSCAR")
    call("pos2aBR")
    st = Structure.from_file("POSCAR_std")
    os.chdir("..")
    shutil.rmtree("symmetrized_st")

    wf = get_wf(st, "/home/tug03990/scripts/read-IR/jengyuan/c2db_ir/irvsp_hse_sp.yaml")
    fws = wf.fws[:3]
    fw_irvsp = IrvspFW(structure=st, parents=fws[-1], additional_fields={"c2db_uid": e["uid"],
                                                                         "spg_c2db": e["spacegroup"],
                                                                         "spg_pymatgen": SpacegroupAnalyzer(st).get_space_group_symbol()
                                                                         })
    fws.append(fw_irvsp)
    wf = Workflow(fws, name=wf.name)

    lpad = LaunchPad.from_file(os.path.expanduser(
        os.path.join("~", "config/project/testIR/irvsp_test/my_launchpad.yaml")))
    wf = clean_up_files(wf, ("WAVECAR*", "CHGCAR*"), wf.fws[-1].name, task_name_constraint=wf.fws[-1].tasks[-1].fw_name)
    wf = add_additional_fields_to_taskdocs(wf, {"c2db_uid": e["uid"]})
    wf = set_execution_options(wf, category="irvsp_test")
    wf = preserve_fworker(wf)
    wf = set_queue_options(wf, walltime="00:30:00", fw_name_constraint=wf.fws[-1].name)
    wf.name = wf.name + ":{}".format(e["uid"])
    lpad.add_wf(wf)
    print(wf)