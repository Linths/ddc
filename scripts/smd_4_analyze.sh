source smd_0_push.sh

python3 analyze_json.py \
	${SMDATA_DIR}/json_filt/${1}.txt ${2}
