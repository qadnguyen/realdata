#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y realdata || :
	@pip install -e .


run_api:
	uvicorn realdata.api.fast:app --reload
