import tempfile
import os
import itertools as it
from nose2.tools import params
import chromosight.cli.chromosight as ccc
from docopt import docopt

COOL = "data_test/example.cool"
BED2D = "data_test/example.bed2"
OUTDIR = tempfile.TemporaryDirectory()
DUMPDIR = tempfile.TemporaryDirectory()
# Select a bunch of arguments that are relevant to test and generate
# all possible combinations. Put them in a datastructure that can be digested
# by docopt
detect_args = ["-I", "--no-plotting", f"-d {DUMPDIR.name}"]
detect_combos = []
for i in range(0, len(detect_args) + 1):
    args_combos = list(it.combinations(detect_args, i))
    for comb in args_combos:
        combo_split = []
        for a in comb:
            for word in a.split(" "):
                combo_split.append(word)
        detect_combos.append(combo_split)


@params(*detect_combos)
def test_cmd_detect(combo):
    """Test for proper exit with different combinations of arguments"""
    args = docopt(ccc.__doc__, ["detect", COOL, OUTDIR.name] + combo)
    ccc.cmd_detect(args)


@params("loops", "borders", "hairpins", "centromeres")
def test_cmd_generate_config(preset):
    """Test for proper exit with different combinations of arguments"""
    tmpconf = tempfile.NamedTemporaryFile(delete=False)
    args = docopt(
        ccc.__doc__, ["generate-config", tmpconf.name, "--preset", preset]
    )
    ccc.cmd_generate_config(args)
    os.unlink(tmpconf.name)


@params("loops", "borders", "hairpins", "centromeres")
def test_cmd_quantify(pattern):
    """Test for proper exit with different combinations of arguments"""
    args = docopt(
        ccc.__doc__,
        ["quantify", "--pattern", pattern, BED2D, COOL, OUTDIR.name],
    )
    ccc.cmd_quantify(args)

def test_cmd_test():
    """Test the ...test command to make sure it doesn't crash."""
    args = docopt(ccc.__doc__, ["test"])
    ccc.cmd_test(args)
