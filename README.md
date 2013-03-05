LDA in python
=============


AUTHOR: fancysimon <fancysimon@gmail.com>


使用
----

### 单机版LDA

	python lda.py -a 0.01 -b 0.1 -k 2 --train_name=train2.txt --compute_likelihood
			--total_iterations=100 --burn_in_iterations=70

### 分布式LDA

[安装MPI和python环境](http://fancysimon.github.com/Programming/2013/03/02/install-mpi-for-python/)

	mpiexec -n 2 -hosts "192.168.1.2,192.168.1.3" python mpi_lda.py -a 0.01 -b 0.1
			--train_name=train2.txt --compute_likelihood --total_iterations=100
			--burn_in_iterations=70

TODO
----

1. Check point for mpi.
2. Comment.
3. Asymmetric alpha(document topic distribution prior).
4. Accumulative model for MPI.
