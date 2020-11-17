from atomate.vasp.fireworks.core import ScanOptimizeFW
from atomate.vasp.powerups import (
    add_additional_fields_to_taskdocs,
    preserve_fworker,
    add_modify_incar,
    add_modify_kpoints,
    set_queue_options,
    set_execution_options,
    clean_up_files
)

from fireworks import LaunchPad, Workflow

from monty.serialization import loadfn


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