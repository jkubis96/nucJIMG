.PHONY: format lint check all

format:
	isort jimg_ncd
	black jimg_ncd

lint:
	pylint --exit-zero --disable=import-error,no-member jimg_ncd

	

all: format lint
