#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y realdata || :
	@pip install -e .


run_api:
	uvicorn prop_value.api.fast:app --reload
