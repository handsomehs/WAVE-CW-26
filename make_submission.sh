#!/bin/bash
set -e

RUN_DIR=$PWD
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

usage="Usage: ./make_submission.sh (-h |-e <exam number> -r <path/to/report>)
Create a submission-ready tarball as aspp-cw1-<exam number>.tar.gz
  -h print this message and exit without doing anything
  -e your exam number, the one that starts with a 'B' and has six numbers after
  -r your report in PDF format please

Both -e and -r options are required to produce a tarball.
"

while [[ $# -gt 0 ]]; do
    case $1 in
    -h)
	echo "$usage"
	exit 0
	shift;;
    -e)
	shift
	if [[ $# -lt 1 ]]; then
	    echo "Missing value for -e"
	fi
	examno=$1
	shift;;
    -r)
	shift
	if [[ $# -lt 1 ]]; then
	    err "Missing value for -e"
	fi
	report=$1
	shift;;
    -*)
	echo "unknown option: $1"
	exit 1;;
    *)
	echo "unexpected argument: $1"
	exit 1;;
    esac
done

if [ -z "$examno" ]; then
    echo "Missing -e <exam number>"
    echo "$usage"
    exit 1
fi

if ! [[ "$examno" =~ ^B[0-9]{6}$ ]]; then
    echo "Invalid exam number: $examno"
    exit 1
else
    echo "Exam number: $examno"
fi

if [ -z "$report" ]; then
    echo "Missing -r <report>"
    echo "$usage"
    exit 1
fi

if [ -f "$report" ]; then
    ft=$(file -b $report)
    if ! [[ "$ft" =~ ^PDF ]]; then
	echo "Report does not appear to be a PDF: $(file $report)"
	exit 1
    fi
else
    echo "Report is not a file: $report"
    exit 1
fi

echo "Report is a PDF"

echo "Checking status of src directory against supplied code..."
# Git hash of the supplied code
refhash=a6a19549b0c3fcfdc691765239c1eadb32103f65
# These files need to be edited
should_change=(wave_cuda.cu wave_omp.cpp)
# These should not be changed to ensure you are testing the same way
# we will
should_not_change=(ASPP.cmake CMakeLists.txt h5io.cpp h5io.h init_sos.cpp init_sos.h main.cpp ndarray.h params.h ufield.cpp ufield.h util.h wave_cpu.cpp wave_cpu.h wave_cuda.h wave_omp.h)

cd "$SCRIPT_DIR"

gitdiff=$(git diff --name-only $refhash -- src)

declare -A changed

if ! [ -z "${gitdiff}" ]; then
    while IFS= read -r change; do
	changed[$change]=1
    done <<< $gitdiff
fi

n_err=0
for nc in "${should_not_change[@]}"; do
    if [ "${changed[src/$nc]}" = 1 ]; then
	echo "WARNING: file that should not be changed has been: $nc"
	n_err=$(($n_err + 1))
	unset changed[src/$nc]
    else
	echo "OK: $nc"
    fi
done

for c in "${should_change[@]}"; do
    if [ "${changed[src/$c]}" = 1 ]; then
	echo "OK: $c"
	unset changed[src/$c]
    else
	echo "WARNING: file that should be changed has not been: $c"
	n_err=$(($n_err + 1))
    fi
done

for c in "${!changed[@]}"; do
    echo "WARNING: unexpected file: $c"
    n_err=$(($n_err + 1))
done

cd src
gitstat=$(git status -s .)
if ! [ -z "${gitdiff}" ]; then
    while IFS= read -r stat; do
	if [[ "$stat" =~ ^'??' ]]; then
	    f=${stat:3}
	    echo "WARNING: unexpected file: $f"
	    n_err=$(($n_err + 1))
	fi
    done <<< $gitstat
fi

if [ $n_err -gt 0 ]; then
    echo "Please consider the errors above before submitting."
    exit 1
fi

echo "Basic check of src directory status OK!"

tmpdir=$(mktemp -d)
echo "Making temporary directory"
mkdir $tmpdir/$examno

echo "Copy report"
cd "$RUN_DIR"
cp $report $tmpdir/$examno/$examno.pdf

echo "Copy source files"
cd "$SCRIPT_DIR/src"
cp "${should_change[@]}" $tmpdir/$examno/

echo "Create tarball"
tarball=aspp-cw1-$examno.tar.gz
cd $tmpdir
tar -czf $tarball $examno

cd "$RUN_DIR"
cp $tmpdir/$tarball ./

echo "Clean up tmpdir"
rm -rf $tmpdir
