source smd_0_push.sh

python3 extract_json.py \
	${SMDATA_DIR}/raw/${1} \
	${SMDATA_DIR}/json_raw/${1} \
	${2}
