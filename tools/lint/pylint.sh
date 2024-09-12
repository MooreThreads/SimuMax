set -euxo pipefail

python3 -m pylint simumax/ --rcfile="$(dirname "$0")"/pylintrc
