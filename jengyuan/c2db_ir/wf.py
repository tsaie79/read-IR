from my_atomate.powerups import add_modify_twod_bs_kpoints
from my_atomate.fireworks.pytopomat import IrvspFW

from atomate.vasp.workflows.base.core import get_wf
from atomate.vasp.database import VaspCalcDb
from atomate.vasp.powerups import (
    add_additional_fields_to_taskdocs,
    preserve_fworker,
    add_modify_incar,
    set_queue_options,
    set_execution_options,
    clean_up_files,
    add_modify_kpoints
)
from fireworks import LaunchPad, Workflow
from pymatgen.core.structure import Structure, SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.sets import MPRelaxSet
from subprocess import call

from mpinterfaces.utils import ensure_vacuum
import os, shutil
import numpy as np

c2db = VaspCalcDb.from_db_file("/home/tug03990/scripts/read-IR/jengyuan/c2db_ir/c2db.json")
for idx, e in enumerate(list(c2db.collection.find({"magstate":"NM"}))[41:42]):
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
    fws = wf.fws
    fw_irvsp = IrvspFW(
        structure=st,
        parents=fws[-1],
        symprec=0.001,
        irvsptodb_kwargs={
            "collection_name": "ir_data",
        }
    )
    fws.append(fw_irvsp)
    wf = Workflow(fws, name=wf.name)
    wf = add_modify_twod_bs_kpoints(
        wf,
        modify_kpoints_params={"kpoints_line_density": 10, "reciprocal_density": 144},
        fw_name_constraint=wf.fws[2].name
    )
    lpad = LaunchPad.from_file(os.path.expanduser(
        os.path.join("~", "config/project/C2DB_IR/calc_data/my_launchpad.yaml")))
    wf = clean_up_files(wf, ("WAVECAR*", "CHGCAR*"), wf.fws[-1].name, task_name_constraint=wf.fws[-1].tasks[-1].fw_name)
    wf = add_additional_fields_to_taskdocs(wf, {"c2db_uid": e["uid"]})
    wf = preserve_fworker(wf)
    for fw_id in [0, 1, 3]:
        wf = set_queue_options(wf, walltime="01:00:00", qos="regular", fw_name_constraint=wf.fws[fw_id].name)
    wf = set_queue_options(wf, walltime="06:00:00", qos="regular", fw_name_constraint=wf.fws[2].name)
    wf.name = wf.name + ":{}".format(e["uid"])

    if 0 == 0:
        wf = set_execution_options(wf, category="calc_data", fworker_name="jyt_cori")
    # else:
    #     wf = set_execution_options(wf, category="calc_data", fworker_name="weiyi_cori")
    lpad.add_wf(wf)
    print(wf)