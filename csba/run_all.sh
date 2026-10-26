set -e
C=$(cd "$(dirname "$0")" && pwd)
cd "${CSBA_ROOT:-work}"
P=$(python -c "import json;print(json.load(open('tables/harm_decision.json'))['rule']['primary'])")
TAGS="$P"; for s in jitter thick; do [ -f preds/${P}_${s}_f4.csv ] && TAGS="$TAGS ${P}_$s"; done
python $C/c05_calibrate.py $TAGS > logs/r_c05.log 2>&1
for s in c06_accuracy c07_contrasts c08_validation c10_geometry c12_atlas c13_marrow c14_tables c15_ablation c20_harmonize c22_harmonization_report c17_tables c18_mainfigs; do
  echo "$s"; python $C/$s.py > logs/r_$s.log 2>&1 || { echo "FAIL $s"; tail -5 logs/r_$s.log; }
done
