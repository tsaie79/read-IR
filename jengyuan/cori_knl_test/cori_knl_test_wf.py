#%%
from atomate.vasp.fireworks import StaticFW
from atomate.vasp.fireworks.jcustom import JHSEStaticFW
from atomate.vasp.jpowerups import scp_files

from atomate.vasp.powerups import *

from fireworks import Workflow, LaunchPad

from pymatgen import Structure, Molecule, Element

import os

mos2 = Structure.from_dict({'@module': 'pymatgen.core.structure',
                                      '@class': 'Structure',
                                      'charge': None,
                                      'lattice': {'matrix': [[3.1840664646845926, 0.0, 0.0],
                                                             [-1.5920332323422963, 2.757482445754964, 0.0],
                                                             [0.0, 0.0, 18.12711264635152]],
                                                  'a': 3.1840664646845926,
                                                  'b': 3.184066464684592,
                                                  'c': 18.12711264635152,
                                                  'alpha': 90.0,
                                                  'beta': 90.0,
                                                  'gamma': 120.00000000000001,
                                                  'volume': 159.15618285810052},
                                      'sites': [{'species': [{'element': 'Mo', 'occu': 1}],
                                                 'abc': [0.0, 0.0, 0.49999999999999994],
                                                 'xyz': [0.0, 0.0, 9.06355632317576],
                                                 'label': 'Mo'},
                                                {'species': [{'element': 'S', 'occu': 1}],
                                                 'abc': [0.6666666666666666, 0.3333333333333333, 0.5862551225713523],
                                                 'xyz': [1.5920332323422963, 0.9191608152516546, 10.62711264635152],
                                                 'label': 'S'},
                                                {'species': [{'element': 'S', 'occu': 1}],
                                                 'abc': [0.6666666666666666, 0.3333333333333333, 0.41374487742864774],
                                                 'xyz': [1.5920332323422963, 0.9191608152516546, 7.5],
                                                 'label': 'S'}]})

# mos2.make_supercell([5,5,1])

for node in [1]:

    mos2 = Molecule(["H"], [[0,0,0]]).get_boxed_structure(5,5,5)

    fw = StaticFW(mos2)
    # fw = JHSEStaticFW(mos2)

    wf = Workflow([fw], name="node:{}".format(node))
    wf = add_additional_fields_to_taskdocs(wf, {"node":"{}".format(node)})

    wf = set_execution_options(wf, category="n{}".format(node))

    wf = preserve_fworker(wf)

    wf = add_modify_incar(wf, {"incar_update": {"NCORE":1, "LAECHG":False, "LVHAR":False,
                                                "LCHARG": False, "LWAVE": False}})

    wf = scp_files(wf, "/home/jengyuantsai/Research/projects/test/")

    lpad = LaunchPad.from_file(os.path.expanduser(os.path.join("~", 'config/project/test/n{}/my_launchpad.yaml'.format(node))))

    lpad.add_wf(wf)
