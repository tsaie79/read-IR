from atomate.vasp.database import VaspCalcDb
from pymatgen.core.structure import Structure
from pathlib import Path
from weiyi.weiyi_read_ir import IrBand

HOME = Path(__file__).absolute().parent.parent
atomate_db = VaspCalcDb.from_db_file('db.json')
example = atomate_db.collection.find_one({'task_label': 'nscf line', 'formula_pretty': 'BN'})
dir_name = example['dir_name'].split('/')[-3:]
OUTPUT_DIR = HOME / '/'.join(dir_name)

struc_dict = example['input']['structure']
struc = Structure.from_dict(struc_dict)

if __name__ == '__main__':
    ir = IrBand(OUTPUT_DIR)
    ir_info = ir.get_band_character()

    # oc = Outcar(OUTPUT_DIR / 'OUTCAR')
    # import gzip
    # import shutil
    #
    # with open('wc', 'wb') as f_out, gzip.open('WAVECAR.gz', 'rb') as f_in:
    #     shutil.copyfileobj(f_in, f_out)
