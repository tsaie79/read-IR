#%%
from atomate.vasp.database import VaspCalcDb
from pymatgen.electronic_structure.plotter import BSPlotter

scan_bs = VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/project/read_ir/read-IR/jengyuan/db_file/scan_bs")
pbe_bs = VaspCalcDb.from_db_file("/Users/jeng-yuantsai/Research/project/read_ir/read-IR/jengyuan/db_file/pbe_bs")
scan_bs = scan_bs.get_band_structure(6)
pbe_bs = pbe_bs.get_band_structure(652)

bs_plotter = BSPlotter(scan_bs)
# bs_plotter.add_bs(scan_bs)
bs_plotter.add_bs(pbe_bs)
plot = bs_plotter.get_plot(vbm_cbm_marker=True)
plot.show()
#%%