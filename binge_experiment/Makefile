.PHONY: optimize
optimize:
	bin/binge_movielens --gpu --xnor optimize --num-iterations 60
	bin/binge_movielens --gpu optimize --num-iterations 60
validate:
	bin/binge_movielens --gpu --xnor validate
	bin/binge_movielens --gpu validate
bench:
	bin/binge_bench
clean:
	rm movielens_100k_validation.log
all: clean optimize validate bench
