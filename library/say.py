import sys

from saying import goodbye

if len(sys.argv) == 2:
    goodbye(sys.argv[1])