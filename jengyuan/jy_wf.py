from atomate.vasp.fireworks.core import ScanOptimizeFW, StaticFW, NonSCFFW, OptimizeFW

from atomate.vasp.workflows.presets.core import wf_bandstructure
from atomate.vasp.powerups import (
    add_additional_fields_to_taskdocs,
    preserve_fworker,
    add_modify_incar,
    add_modify_kpoints,
    set_queue_options,
    set_execution_options,
    clean_up_files,
    modify_gzip_vasp
)
from atomate.vasp.database import VaspCalcDb

from pymatgen import Structure
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.sets import MPMetalRelaxSet

from fireworks import LaunchPad, Workflow

from monty.serialization import loadfn

import os, glob

from weiyi.modify_poscar import modify


def relax_pc():
    lpad = LaunchPad.from_file("/home/tug03990/atomate/example/config/project/"
                               "symBaseBinaryQubit/scan_relax_pc/my_launchpad.yaml")

    mx2s = loadfn("/home/tug03990/atomate/example/config/project/symBaseBinaryQubit/"
                  "scan_relax_pc/gap_gt1-binary-NM.json")

    for mx2 in mx2s:
        if mx2["irreps"] and mx2["formula"] == "Rh2Br6" and mx2["spacegroup"] == "P3":
            pc = mx2["structure"]
            scan_opt = ScanOptimizeFW(structure=pc, name="SCAN_relax")
            wf = Workflow([scan_opt], name="{}:SCAN_opt".format(mx2["formula"]))
            wf = add_modify_incar(wf)
            wf = add_modify_incar(
                wf,
                {
                    "incar_update": {
                        "LCHARG": False,
                        "LWAVE": False
                    }
                }
            )
            mx2.pop("structure")
            wf = add_additional_fields_to_taskdocs(
                wf,
                {"c2db_info": mx2}
            )
            wf = add_modify_incar(wf)
            wf = set_execution_options(wf, category="scan_relax_pc")
            wf = preserve_fworker(wf)
            lpad.add_wf(wf)


def test_IR(cat="genWavecar"):
    lpad = LaunchPad.from_file(
        os.path.join(
            os.path.expanduser("~"),
            "config/project/testIR/{}/my_launchpad.yaml".format(cat)))
    col = VaspCalcDb.from_db_file(
        os.path.join(
            os.path.expanduser("~"),
            "config/project/symBaseBinaryQubit/scan_relax_pc/db.json")).collection

    input_st = Structure.from_dict(col.find_one({"chemsys":"Mo-S"})["output"]["structure"])

    wf = wf_bandstructure(input_st)
    for fw_name in ["static", "nscf"]:
        wf = add_modify_incar(wf, {"incar_update":{"LWAVE":True, "ISYM":2}}, fw_name_constraint=fw_name)

    for ispin in [1,2]:
        wf = add_modify_incar(wf, {"incar_update":{"ISPIN":ispin}})
        wf = add_modify_incar(wf)
        wf = set_execution_options(wf, category=cat)
        wf = preserve_fworker(wf)
        wf = set_queue_options(wf, "01:00:00")
        if ispin == 1:
            wf.name = "MoS2_spin_{}_2D_k".format("off")
        elif ispin == 2:
            wf.name = "MoS2_spin_{}_2D_k".format("on")
        lpad.add_wf(wf)


def ML_bs_wf(metal=True):
    cat = None
    if metal:
        cat = "metal"
    else:
        cat = "nonmetal"

    def bs_fws(structure):
        opt = OptimizeFW(structure=structure)
        static_fw = StaticFW(structure=structure, parents=opt)
        line_fw = NonSCFFW(structure=structure,
                           mode="line",
                           parents=static_fw
                           )

        wf = Workflow([opt, static_fw, line_fw], name="{}:pbe_bs".format(structure.formula))

        updates = {
            "NELM": 150,
            "EDIFF": 1E-5,
            "ISPIN": 1,
            "LAECHG": False,
            "LASPH": False,
            "LREAL": "Auto",
            "SIGMA": 0.05
        }

        wf = add_modify_incar(wf, {"incar_update": updates})
        wf = add_modify_incar(wf, {"incar_update": {"LCHARG":False, "ISIF":3, "EDIFFG":-0.01, "EDIFF":1E-6}}, opt.name)
        wf = add_modify_incar(wf, {"incar_update": {"LCHARG":True, "LVHAR":True}}, static_fw.name)
        wf = add_modify_incar(wf, {"incar_update": {"LWAVE":True, "LCHARG":False, "ISYM":2}}, line_fw.name)
        wf = modify_gzip_vasp(wf, False)
        wf = clean_up_files(wf, files=["CHG*", "DOS*", "LOCPOT*"], fw_name_constraint=line_fw.name,
                            task_name_constraint="VaspToDb")
        return wf

    def bs_fws_metal(structure):
        opt = OptimizeFW(structure=structure, vasp_input_set=MPMetalRelaxSet(structure))
        static_fw = StaticFW(structure=structure, parents=opt)
        line_fw = NonSCFFW(structure=structure,
                           mode="line",
                           parents=static_fw
                           )

        wf = Workflow([opt, static_fw, line_fw], name="{}:pbe_bs".format(structure.formula))

        updates = {
            "NELM": 150,
            "EDIFF": 1E-5,
            "ISPIN": 1,
            "LAECHG": False,
            "LASPH": False,
            "LREAL": "Auto",
            "ISMEAR": 1,
            "SIGMA": 0.05,
        }

        wf = add_modify_incar(wf, {"incar_update": updates})
        wf = add_modify_incar(wf, {"incar_update": {"LCHARG":False, "ISIF":3, "EDIFFG":-0.01, "EDIFF":1E-6}}, opt.name)
        wf = add_modify_incar(wf, {"incar_update": {"LCHARG":True, "LVHAR":True}}, static_fw.name)
        kpt = Kpoints.automatic_density_by_vol(structure, 200)
        wf = add_modify_kpoints(wf, {"kpoints_update":{"kpts": kpt.kpts}}, static_fw.name)
        wf = add_modify_incar(wf, {"incar_update": {"LWAVE":True, "LCHARG":False, "ISYM":2}}, line_fw.name)
        wf = modify_gzip_vasp(wf, False)
        wf = clean_up_files(wf, files=["CHG*", "DOS*", "LOCPOT*"], fw_name_constraint=line_fw.name,
                            task_name_constraint="VaspToDb")
        return wf

    lpad = LaunchPad.from_file(
        os.path.expanduser(os.path.join("~", "config/project/ML_data/PBE_bulk/my_launchpad.yaml")))

    base_dir = "/project/projectdirs/m2663/tsai/ML_data/PBE_bulk"
    wf_func = None
    if metal:
        p = "cifs_metal_modified/*"
        wf_func = bs_fws_metal
    else:
        p = "cifs_nonmetal_modified/*"
        wf_func = bs_fws
    for st in glob.glob(os.path.join(base_dir, p))[500:]:
        print(st)
        input_st = Structure.from_file(st)
        mod_st = modify(input_st)
        if mod_st:
            input_st = mod_st
        wf = wf_func(input_st)
        wf = add_modify_incar(wf)
        wf = set_execution_options(wf, category=cat)
        wf = preserve_fworker(wf)

        wf = add_additional_fields_to_taskdocs(
            wf,
            {
                "mp_id": st.split("/")[-1],
                "wfs": [fw.name for fw in wf.fws],
                "material_type": cat
            }
        )
        wf.name = wf.name+":{}".format(st.split("/")[-1])
        lpad.add_wf(wf)



if __name__ == '__main__':
    ML_bs_wf(metal=False)