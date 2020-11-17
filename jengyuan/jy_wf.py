from atomate.vasp.fireworks.core import ScanOptimizeFW, StaticFW, NonSCFFW
from atomate.vasp.workflows.presets.core import wf_bandstructure
from atomate.vasp.powerups import (
    add_additional_fields_to_taskdocs,
    preserve_fworker,
    add_modify_incar,
    add_modify_kpoints,
    set_queue_options,
    set_execution_options,
    clean_up_files
)
from atomate.vasp.database import VaspCalcDb

from pymatgen import Structure

from fireworks import LaunchPad, Workflow

from monty.serialization import loadfn

import os


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


def bs_scan_wf(cat="bs_ir"):

    def wf(structure):
        static_fw = StaticFW(structure=structure)
        line_fw = NonSCFFW(structure=structure, mode="line", parents=static_fw)

        wf = Workflow([static_fw, line_fw], name="{}:read_ir".format(structure.formula))

        updates = {
            "ADDGRID": True,
            "LASPH": True,
            "LDAU": False,
            "LMIXTAU": True,
            "METAGGA": "SCAN",
            "NELM": 200,
            "EDIFF": 1E-5,
        }

        wf = add_modify_incar(wf, {"incar_update": updates})
        wf = add_modify_incar(wf, {"incar_update": {"LCHARG":True, "LVHAR":True}}, static_fw.name)
        wf = add_modify_incar(wf, {"incar_update": {"LWAVE":True, "LCHARG":False}}, line_fw.name)
        wf = clean_up_files(wf, files=["CHG*", "DOS*", "LOCPOT*"], fw_name_constraint=line_fw.name,
                            task_name_constraint="VaspToDb")
        return wf

    lpad = LaunchPad.from_file(
        os.path.join(
            os.path.expanduser("~"),
            "config/project/read_ir/{}/my_launchpad.yaml".format(cat)))
    col = VaspCalcDb.from_db_file(
        os.path.join(
            os.path.expanduser("~"),
            "config/project/symBaseBinaryQubit/scan_relax_pc/db.json")).collection

    for e in col.find():
        input_st = Structure.from_dict(e["output"]["structure"])
        wf = wf(input_st)

        wf = add_modify_incar(wf)
        wf = set_execution_options(wf, category=cat)
        wf = preserve_fworker(wf)

        wf = add_additional_fields_to_taskdocs(
            wf,
            {
                "pc_from": "symBaseBinaryQubit/scan_relax_pc/{}".format(e["task_id"]),
                "c2db_info": e["c2db_info"]
            }
        )

        lpad.add_wf(wf)



