# Add additional .c files here if you added any yourself.
ADDITIONAL_SOURCES =

# Add additional .h files here if you added any yourself.
ADDITIONAL_HEADERS =

# -- Do not modify below this point - will get replaced during testing --

SOURCES = kvstore.c hash.c  parser.c  request_dispatcher.c  server_utils.c \
		  $(ADDITIONAL_SOURCES) intercept.c
HEADERS = kvstore.h common.h  hash.h  parser.h server_utils.h  request_dispatcher.h \
		  $(ADDITIONAL_HEADERS) intercept.h
META = Makefile check.py test_client.py README.rst

CFLAGS = -Og -ggdb -std=gnu99 -Wall -Wextra -fsanitize=address \
		 -fno-omit-frame-pointer -pthread
LDFLAGS = -fsanitize=address -pthread -ldl

DOCKERIMG = vusec/vu-os-kvstore-check
TARGET= kvstore

.PHONY: all tarball clean check

all: kvstore

tarball: kvstore.tar.gz

kvstore.tar.gz: $(SOURCES) $(HEADERS) $(META)
	tar czf $@ $^

check:
	@./check.py

docker-update:
	docker pull $(DOCKERIMG)

docker-check: kvstore.tar.gz
	mkdir -p docker_mnt
	cp $^ docker_mnt/
	docker run -i -t --rm -v "`pwd`/docker_mnt:/submission" $(DOCKERIMG) /test

docker-run:
	docker run --privileged -it --rm  -v "`pwd`:/code" -w /code \
	--name os_$(TARGET) $(DOCKERIMG) /bin/bash

docker-connect:
# 	If the docker is not running, throw an error
	@if ! docker container inspect os_$(TARGET) > /dev/null 2>&1; then \
		echo "\033[0;31mDocker container os_$(TARGET) is not running! Run \"make docker-run\" first in another terminal.\033[0m" && false; \
	fi

# 	Connect to the docker container
	docker container exec -it os_$(TARGET) /bin/bash

kvstore: $(SOURCES:.c=.o)
	$(CC) -o $@ $^ $(LDFLAGS)

$(SOURCES:.c=.o): $(HEADERS)

clean:
	rm -f kvstore *.o
