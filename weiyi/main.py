from weiyi.read_ir_new import BandCharacter
from pathlib import Path
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from atomate.vasp.database import VaspCalcDb


def get_character(file_dir: str):
    file_dir = Path(file_dir)
    struc = Structure.from_file(file_dir/'POSCAR')
    sg_number = SpacegroupAnalyzer(struc, symprec=0.1).get_space_group_number()
    bc = BandCharacter(file_dir, sg_number)
    character = bc.get_band_character()
    print(character)


def get_character_from_db():
    HOME = Path(__file__).absolute().parent.parent
    atomate_db = VaspCalcDb.from_db_file('db.json')
    example = atomate_db.collection.find_one({'task_label': 'nscf line'})
    OUTPUT_DIR = HOME / '/'.join(example['dir_name'].split('/')[-3:])

    sg_number = example['output']['spacegroup']['number']
    bc = BandCharacter(OUTPUT_DIR, sg_number)
    char = bc.get_band_character()
    print(char)


if __name__ == '__main__':
    get_character('example')
    get_character_from_db()
