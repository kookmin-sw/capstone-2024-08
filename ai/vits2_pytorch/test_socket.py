
# NOTE [ DataLoader on Linux and open files limit ]
#
# On Linux when DataLoader is used with multiprocessing we pass the data between
# the root process and the workers through SHM files. We remove those files from
# the filesystem as soon as they are created and keep them alive by
# passing around their file descriptors through AF_UNIX sockets. (See
# docs/source/multiprocessing.rst and 'Multiprocessing Technical Notes` in
# the wiki (https://github.com/pytorch/pytorch/wiki).)
#
# This sometimes leads us to exceeding the open files limit. When that happens,
# and the offending file descriptor is coming over a socket, the `socket` Python
# package silently strips the file descriptor from the message, setting only the
# `MSG_CTRUNC` flag (which might be a bit misleading since the manpage says that
# it _indicates that some control data were discarded due to lack of space in
# the buffer for ancillary data_). This might reflect the C implementation of
# AF_UNIX sockets.
#
# This behaviour can be reproduced with the script and instructions at the
# bottom of this note.
#
# When that happens, the standard Python `multiprocessing` (and not
# `torch.multiprocessing`) raises a `RuntimeError: received 0 items of ancdata`
#
# Sometimes, instead of the FD being stripped, you may get an `OSError:
# Too many open files`, both in the script below and in DataLoader. However,
# this is rare and seems to be nondeterministic.
#

#!/usr/bin/env python3
import sys
import socket
import os
import array
import shutil
import socket


if len(sys.argv) != 4:
    print("Usage: ", sys.argv[0], " tmp_dirname iteration (send|recv)")
    sys.exit(1)

if __name__ == '__main__':
    dirname = sys.argv[1]
    sock_path = dirname + "/sock"
    iterations = int(sys.argv[2])
    def dummy_path(i):
        return dirname + "/" + str(i) + ".dummy"


    if sys.argv[3] == 'send':
        while not os.path.exists(sock_path):
            pass
        client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        client.connect(sock_path)
        for i in range(iterations):
            fd = os.open(dummy_path(i), os.O_WRONLY | os.O_CREAT)
            ancdata = array.array('i', [fd])
            msg = bytes([i % 256])
            print("Sending fd ", fd, " (iteration #", i, ")")
            client.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, ancdata)])


    else:
        assert sys.argv[3] == 'recv'

        if os.path.exists(dirname):
            raise Exception("Directory exists")

        os.mkdir(dirname)

        print("Opening socket...")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        server.bind(sock_path)

        print("Listening...")
        for i in range(iterations):
            a = array.array('i')
            msg, ancdata, flags, addr = server.recvmsg(1, socket.CMSG_SPACE(a.itemsize))
            assert(len(ancdata) == 1)
            cmsg_level, cmsg_type, cmsg_data = ancdata[0]
            a.frombytes(cmsg_data)
            print("Received fd ", a[0], " (iteration #", i, ")")

        shutil.rmtree(dirname)

# Steps to reproduce:
#
# 1. Run two shells and set lower file descriptor limit in the receiving one:
# (shell1) ulimit -n 1020
# (shell2) ulimit -n 1022
#
# 2. Run the script above with the `recv` option in the first shell
# (shell1) ./test_socket.py sock_tmp 1017 recv
#
# 3. Run the script with the `send` option in the second shell:
# (shell2) ./test_socket.py sock_tmp 1017 send