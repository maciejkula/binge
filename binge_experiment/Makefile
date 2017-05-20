.PHONY: optimize
optimize:
	bin/binge_movielens --gpu optimize --num-iterations 50
	bin/binge_movielens --gpu --xnor optimize --num-iterations 50
validate:
	bin/binge_movielens --gpu validate
	bin/binge_movielesn --gpu --xnor validate
clean:
	rm movielens_100k_validation.log
all: clean optimize validate
