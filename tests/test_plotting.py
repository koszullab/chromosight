from nose2.tools import params
import chromosight.utils.plotting as cup
import chromosight.utils.detection as cud


mat, chroms, bins, res = cio.load_cool("data_test/example.cool")
mat = mat.tocsr()
#  Get all intra-chromosomal matrices
intra_mats = [
    mat[s:e, s:e] for s, e in zip(chroms["start_bin"], chroms["end_bin"])
]
pattern_list = []
window_list = []


@params(*zip(intra_mats, pattern_list))
def test_pattern_plot(contact_map, patterns):
    cup.pattern_plot(contact_map, patterns, output=None, name=None)


def test_distance_plot():
    cup.distance_plot(intra_mats, labels=None)


@params(*window_list)
def test_pileup_plot(windows):
    pileup_pattern = cud.pileup_patterns(windows)
    cup.pileup_plot(pileup_pattern, name="pileup patterns", output=None)


@params(*zip(pattern_list, chroms.name))
def test_plot_whole_matrix(patterns, chrom):
    region = chroms.loc[chroms.name == chrom, ["start_bin", "end_bin"]]
    cup.plot_whole_matrix(mat, patterns, out=None, region=region)
