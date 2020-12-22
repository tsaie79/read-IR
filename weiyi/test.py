from atomate.vasp.database import VaspCalcDb
from pathlib import Path
from weiyi.read_ir_new import BandCharacter
from weiyi.modify_poscar import SG_INV

HOME = Path(__file__).absolute().parent.parent
atomate_db = VaspCalcDb.from_db_file('db.json')
examples = atomate_db.collection.find({'task_label': 'nscf line',
                                       # 'output.spacegroup.number': {'$in': SG_INV.tolist()}
                                       })
example = examples[0]
OUTPUT_DIR = HOME / '/'.join(example['dir_name'].split('/')[-3:])
sg_number = example['output']['spacegroup']['number']

bc = BandCharacter(OUTPUT_DIR, sg_number)
char = bc.get_band_character()
print(char)
