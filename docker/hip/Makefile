.PHONY:
all: ava-rocm

ava-rocm: .Dockerfile.ava-rocm.stamp

.Dockerfile.%.stamp: Dockerfile.%
	docker build -t $* -f $^ .
	touch $@

.PHONY:
clean:
	rm .Dockerfile.*.stamp -f